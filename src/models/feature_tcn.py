import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)

        return self.relu(out + residual)


class FeatureTCNModel:
    def __init__(
        self,
        input_size,
        num_channels=(32, 32),
        kernel_size=3,
        dropout=0.1,
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        device=None
    ):
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        layers = []
        in_channels = input_size

        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers).to(self.device)
        self.fc = nn.Linear(num_channels[-1], 1).to(self.device)

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.fc.parameters()),
            lr=learning_rate
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, seq_len, n_features = X_train.shape

        X_flat = X_train.reshape(-1, n_features)

        self.x_scaler.fit(X_flat)
        self.y_scaler.fit(y_train.reshape(-1, 1))

        X_scaled = self.x_scaler.transform(X_flat).reshape(
            n_samples,
            seq_len,
            n_features
        )

        y_scaled = self.y_scaler.transform(y_train.reshape(-1, 1))

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

        has_validation = X_val is not None and y_val is not None

        if has_validation:
            n_val = X_val.shape[0]

            X_val_scaled = self.x_scaler.transform(
                X_val.reshape(-1, n_features)
            ).reshape(n_val, seq_len, n_features)

            y_val_scaled = self.y_scaler.transform(y_val.reshape(-1, 1))

            X_val_tensor = torch.tensor(
                X_val_scaled,
                dtype=torch.float32
            ).to(self.device)

            y_val_tensor = torch.tensor(
                y_val_scaled,
                dtype=torch.float32
            ).to(self.device)

        self.history_ = []

        for epoch in range(self.epochs):
            self.network.train()
            self.fc.train()

            total_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # TCN expects: batch, channels, sequence
                X_batch = X_batch.transpose(1, 2)

                self.optimizer.zero_grad()

                out = self.network(X_batch)

                last_state = out[:, :, -1]

                preds = self.fc(last_state)

                loss = self.loss_fn(preds, y_batch)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)

            if has_validation:
                self.network.eval()
                self.fc.eval()

                with torch.no_grad():
                    Xv = X_val_tensor.transpose(1, 2)
                    val_out = self.network(Xv)
                    val_last = val_out[:, :, -1]
                    val_preds = self.fc(val_last)
                    val_loss = self.loss_fn(val_preds, y_val_tensor).item()
            else:
                val_loss = None

            self.history_.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

        return self

    def predict(self, X_test):
        n_samples, seq_len, n_features = X_test.shape

        X_scaled = self.x_scaler.transform(
            X_test.reshape(-1, n_features)
        ).reshape(n_samples, seq_len, n_features)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).to(self.device)

        self.network.eval()
        self.fc.eval()

        with torch.no_grad():
            X_tensor = X_tensor.transpose(1, 2)
            out = self.network(X_tensor)
            last_state = out[:, :, -1]
            preds_scaled = self.fc(last_state)

        preds = self.y_scaler.inverse_transform(
            preds_scaled.cpu().numpy()
        )

        return preds.flatten()