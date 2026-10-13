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


class FeatureCausalTransformerModel:

    def __init__(
        self,
        input_size,
        residual=False,
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
        self.residual = residual
        self.d_model = d_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.input_projection = nn.Linear(input_size, d_model).to(self.device)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 512, d_model, device=self.device)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        ).to(self.device)

        self.fc = nn.Linear(d_model, 1).to(self.device)

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            list(self.input_projection.parameters())
            + list(self.encoder.parameters())
            + [self.pos_embedding]
            + list(self.fc.parameters()),
            lr=learning_rate
        )

    def _causal_mask(self, seq_len):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device),
            diagonal=1
        ).bool()

        return mask

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Multi-feature causal transformer training.

        Saves training history in:
        self.history_
        """

        n_samples, seq_len, n_features = X_train.shape

        X_flat = X_train.reshape(-1, n_features)
        self.x_scaler.fit(X_flat)
        self.y_scaler.fit(y_train.reshape(-1, 1))

        X_scaled = self.x_scaler.transform(X_flat).reshape(
            n_samples,
            seq_len,
            n_features
        )

        y_scaled = self.y_scaler.transform(
            y_train.reshape(-1, 1)
        )

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

        # Optional validation data
        if X_val is not None and y_val is not None:
            n_val, _, _ = X_val.shape

            X_val_scaled = self.x_scaler.transform(
                X_val.reshape(-1, n_features)
            ).reshape(n_val, seq_len, n_features)

            y_val_scaled = self.y_scaler.transform(
                y_val.reshape(-1, 1)
            )

            X_val_tensor = torch.tensor(
                X_val_scaled,
                dtype=torch.float32
            ).to(self.device)

            y_val_tensor = torch.tensor(
                y_val_scaled,
                dtype=torch.float32
            ).to(self.device)

            has_validation = True
        else:
            X_val_tensor = None
            y_val_tensor = None
            has_validation = False

        self.input_projection.train()
        self.encoder.train()
        self.fc.train()

        mask = self._causal_mask(seq_len)

        history = []

        for epoch in range(self.epochs):

            total_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                tokens = self.input_projection(X_batch)
                tokens = tokens + self.pos_embedding[:, :seq_len, :]

                out = self.encoder(tokens, mask=mask)

                last_token = out[:, -1, :]

                preds = self.fc(last_token)

                if self.residual:
                    last_x_scaled = X_batch[:, -1, 0:1]
                    preds = last_x_scaled + preds

                loss = self.loss_fn(preds, y_batch)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)

            if has_validation:
                self.input_projection.eval()
                self.encoder.eval()
                self.fc.eval()

                with torch.no_grad():
                    val_tokens = self.input_projection(X_val_tensor)
                    val_tokens = val_tokens + self.pos_embedding[:, :seq_len, :]

                    val_out = self.encoder(val_tokens, mask=mask)

                    val_last_token = val_out[:, -1, :]

                    val_preds = self.fc(val_last_token)

                    if self.residual:
                        val_last_x_scaled = X_val_tensor[:, -1, 0:1]
                        val_preds = val_last_x_scaled + val_preds

                    val_loss = self.loss_fn(
                        val_preds,
                        y_val_tensor
                    ).item()

                self.input_projection.train()
                self.encoder.train()
                self.fc.train()
            else:
                val_loss = None

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            if (epoch + 1) % 10 == 0:
                if has_validation:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs}, "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs}, "
                        f"Train Loss: {train_loss:.6f}"
                    )

        self.history_ = history

        return self

    def predict(self, X_test):

        n_samples, seq_len, n_features = X_test.shape

        X_flat = X_test.reshape(-1, n_features)

        X_scaled = self.x_scaler.transform(X_flat).reshape(
            n_samples,
            seq_len,
            n_features
        )

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).to(self.device)

        self.input_projection.eval()
        self.encoder.eval()
        self.fc.eval()

        mask = self._causal_mask(seq_len)

        with torch.no_grad():

            tokens = self.input_projection(X_tensor)
            tokens = tokens + self.pos_embedding[:, :seq_len, :]

            out = self.encoder(tokens, mask=mask)

            last_token = out[:, -1, :]

            preds_scaled = self.fc(last_token)

            if self.residual:
                last_x_scaled = X_tensor[:, -1, 0:1]
                preds_scaled = last_x_scaled + preds_scaled

        preds = self.y_scaler.inverse_transform(
            preds_scaled.cpu().numpy()
        )

        return preds.flatten()