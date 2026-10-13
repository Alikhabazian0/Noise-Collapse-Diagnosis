import math
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


class PatchTSTLevelModel:

    def __init__(
        self,
        input_size=1,
        patch_len=4,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        epochs=50,
        batch_size=32,
        learning_rate=0.0003,
        device=None
    ):
        self.input_size = input_size
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()

        self.patch_projection = nn.Linear(
            patch_len * input_size,
            d_model
        ).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        ).to(self.device)

        self.fc = nn.Linear(d_model, 1).to(self.device)

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            list(self.patch_projection.parameters())
            + list(self.transformer.parameters())
            + list(self.fc.parameters()),
            lr=learning_rate
        )

    def _create_patches(self, X):
        """
        X shape: (batch, seq_len, input_size)

        Returns:
        patches shape: (batch, num_patches, patch_len * input_size)
        """
        patches = []

        seq_len = X.shape[1]

        for start in range(0, seq_len - self.patch_len + 1, self.stride):
            end = start + self.patch_len
            patch = X[:, start:end, :]
            patch = patch.reshape(X.shape[0], -1)
            patches.append(patch)

        return torch.stack(patches, dim=1)

    def fit(self, X_train, y_train):
        """
        system level prediction:

        [x_{t-15}, ..., x_t] -> x_{t+1}
        """

        train_values = np.concatenate([
            X_train.flatten(),
            y_train.reshape(-1)
        ]).reshape(-1, 1)

        self.scaler.fit(train_values)

        X_scaled = self.scaler.transform(
            X_train.reshape(-1, 1)
        ).reshape(X_train.shape[0], X_train.shape[1], 1)

        y_scaled = self.scaler.transform(
            y_train.reshape(-1, 1)
        )

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.patch_projection.train()
        self.transformer.train()
        self.fc.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                patches = self._create_patches(X_batch)

                tokens = self.patch_projection(patches)

                out = self.transformer(tokens)

                last_token = out[:, -1, :]

                preds = self.fc(last_token)

                loss = self.loss_fn(preds, y_batch)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {total_loss / len(loader):.6f}"
                )

        return self

    def predict(self, X_test):
        
        X_scaled = self.scaler.transform(
            X_test.reshape(-1, 1)
        ).reshape(X_test.shape[0], X_test.shape[1], 1)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).to(self.device)

        self.patch_projection.eval()
        self.transformer.eval()
        self.fc.eval()

        with torch.no_grad():
            patches = self._create_patches(X_tensor)
            tokens = self.patch_projection(patches)
            out = self.transformer(tokens)
            last_token = out[:, -1, :]
            preds_scaled = self.fc(last_token)

        preds = self.scaler.inverse_transform(
            preds_scaled.cpu().numpy()
        )

        return preds.flatten()