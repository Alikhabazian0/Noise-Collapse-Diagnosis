import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_channels, out_channels, kernel_size, 
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout()
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res  = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
class TCNModel:
    def __init__(
            self,
            input_size=1,
            channels=(32,32),
            kernel_size=3,
            dropout=0.1,
            epochs=30,
            batch_size=32,
            learning_rate=0.001,
            device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()

        layers = []
        for i, out_channels in enumerate(channels):
            in_channels = input_size if i == 0 else channels[i - 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )

        self.tcn = nn.Sequential(*layers).to(self.device)
        self.fc = nn.Linear(channels[-1], 1).to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            list(self.tcn.parameters()) + list(self.fc.parameters()),
            lr=learning_rate
        )

    def fit(self, X_train, y_train):
        train_values = np.concatenate([X_train.flatten(), y_train]).reshape(-1, 1)
        self.scaler.fit(train_values)

        X_scaled = self.scaler.transform(
            X_train.reshape(-1, 1)
        ).reshape(X_train.shape[0], X_train.shape[1], 1)

        y_scaled = self.scaler.transform(y_train.reshape(-1, 1))

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size = self.batch_size,
            shuffle=False
        )

        self.tcn.train()
        self.fc.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                out = self.tcn(X_batch)
                last_hidden = out[:, :, -1]
                preds = self.fc(last_hidden)

                loss = self.loss_fn(preds, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, loss: {total_loss / len(loader):.6f}")
        
        return self
    
    def predict(self, X_test):
        X_scaled = self.scaler.transform(
            X_test.reshape(-1,1)
        ).reshape(X_test.shape[0], X_test.shape[1], 1)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1).to(self.device)

        self.tcn.eval()
        self.fc.eval()

        with torch.no_grad():
            out = self.tcn(X_tensor)
            last_hidden = out[:, :, -1]
            preds_scaled = self.fc(last_hidden)

        preds = self.scaler.inverse_transform(preds_scaled.cpu().numpy())

        return preds.flatten()
    