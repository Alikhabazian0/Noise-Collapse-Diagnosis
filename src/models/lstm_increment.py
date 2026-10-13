import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class LSTMIncrementModel:

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1,
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)

        self.fc = nn.Linear(hidden_size, 1).to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.fc.parameters()),
            lr=learning_rate
        )

    def fit(self, X_train, y_train):
        last_values = X_train[:, -1]
        y_increment = y_train - last_values

        self.x_scaler.fit(X_train.reshape(-1, 1))
        self.y_scaler.fit(y_increment.reshape(-1, 1))

        X_scaled = self.x_scaler.transform(
            X_train.reshape(-1, 1)
        ).reshape(X_train.shape[0], X_train.shape[1], 1)

        y_scaled = self.y_scaler.transform(
            y_increment.reshape(-1, 1)
        )

        dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32)
        )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.train()
        self.fc.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                output, _ = self.model(X_batch)
                last_hidden = output[:, -1, :]
                preds = self.fc(last_hidden)

                loss = self.loss_fn(preds, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(loader):.6f}")

        return self

    def predict(self, X_test):
        last_values = X_test[:, -1]

        X_scaled = self.x_scaler.transform(
            X_test.reshape(-1, 1)
        ).reshape(X_test.shape[0], X_test.shape[1], 1)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        self.fc.eval()

        with torch.no_grad():
            output, _ = self.model(X_tensor)
            last_hidden = output[:, -1, :]
            pred_increment_scaled = self.fc(last_hidden)

        pred_increment = self.y_scaler.inverse_transform(
            pred_increment_scaled.cpu().numpy()
        ).flatten()

        return last_values + pred_increment