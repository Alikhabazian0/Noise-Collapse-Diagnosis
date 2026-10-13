import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class LSTMModel:

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
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()
        
        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)

        self.fc = nn.Linear(hidden_size, 1).to(self.device)

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.fc.parameters()),
            lr = self.learning_rate
        )

    def fit(self, X_train, y_train):

        train_values = np.concatenate([
            X_train.flatten(),
            y_train.reshape(-1)
        ]).reshape(-1, 1)

        self.scaler.fit(train_values)
        
        X_scaled = self.scaler.transform(
            X_train.reshape(-1,1)
        ).reshape(X_train.shape[0], X_train.shape[1], 1)
        
        y_scaled = self.scaler.transform(
            y_train.reshape(-1,1)
        )
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)

        loader  = DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle = False
        )

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
                avg_loss = total_loss / len(loader)
                print(f"Epochs {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        return self
    
    def predict(self, X_test):
        
        X_scaled = self.scaler.transform(
            X_test.reshape(-1,1)
        ).reshape(X_test.shape[0], X_test.shape[1], 1)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        self.fc.eval()

        with torch.no_grad():
            output, _ = self.model(X_tensor)
            last_hidden = output[:, -1, :]
            preds_scaled = self.fc(last_hidden)
        
        preds = self.scaler.inverse_transform(
            preds_scaled.cpu().numpy()
        )

        return preds.flatten()