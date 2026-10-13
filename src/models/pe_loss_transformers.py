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


def soft_ordinal_entropy_loss(y_true, y_pred, order=5, tau=0.1, eps=1e-8):
    """
    Differentiable soft ordinal-complexity mismatch.

    Uses local pairwise ordering probabilities inside the batch.
    This is not exact permutation entropy, because exact PE is non-differentiable.
    It is a differentiable PE-inspired surrogate.
    """

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    n = y_pred.shape[0]

    if n < order + 1:
        return torch.tensor(0.0, device=y_pred.device)

    true_windows = y_true.unfold(0, order, 1)
    pred_windows = y_pred.unfold(0, order, 1)

    def soft_entropy(windows):
        features = []

        for i in range(order):
            for j in range(i + 1, order):
                diff = windows[:, j] - windows[:, i]
                prob = torch.sigmoid(diff / tau)
                features.append(prob)

        probs = torch.stack(features, dim=1)

        p = probs.mean(dim=0)
        entropy = -torch.sum(
            p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)
        )

        entropy = entropy / len(features)

        return entropy

    true_entropy = soft_entropy(true_windows).detach()
    pred_entropy = soft_entropy(pred_windows)

    return torch.abs(true_entropy - pred_entropy)


class PELossCausalTransformerModel:

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
        pe_lambda=0.05,
        pe_order=5,
        pe_tau=0.1,
        device=None
    ):
        self.input_size = input_size
        self.residual = residual
        self.d_model = d_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.pe_lambda = pe_lambda
        self.pe_order = pe_order
        self.pe_tau = pe_tau

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

        self.mse_loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            list(self.input_projection.parameters())
            + list(self.encoder.parameters())
            + [self.pos_embedding]
            + list(self.fc.parameters()),
            lr=learning_rate
        )

    def _causal_mask(self, seq_len):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=self.device),
            diagonal=1
        ).bool()

    def _forward_scaled(self, X_batch):
        seq_len = X_batch.shape[1]

        tokens = self.input_projection(X_batch)
        tokens = tokens + self.pos_embedding[:, :seq_len, :]

        mask = self._causal_mask(seq_len)

        out = self.encoder(tokens, mask=mask)
        last_token = out[:, -1, :]

        preds = self.fc(last_token)

        if self.residual:
            last_x_scaled = X_batch[:, -1, 0:1]
            preds = last_x_scaled + preds

        return preds

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

            self.input_projection.train()
            self.encoder.train()
            self.fc.train()

            total_loss = 0.0
            total_mse = 0.0
            total_pe = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                preds = self._forward_scaled(X_batch)

                mse = self.mse_loss(preds, y_batch)

                pe_penalty = soft_ordinal_entropy_loss(
                    y_true=y_batch,
                    y_pred=preds,
                    order=self.pe_order,
                    tau=self.pe_tau
                )

                loss = mse + self.pe_lambda * pe_penalty

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_mse += mse.item()
                total_pe += pe_penalty.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)
            train_mse = total_mse / max(n_batches, 1)
            train_pe = total_pe / max(n_batches, 1)

            if has_validation:
                self.input_projection.eval()
                self.encoder.eval()
                self.fc.eval()

                with torch.no_grad():
                    val_preds = self._forward_scaled(X_val_tensor)

                    val_mse = self.mse_loss(
                        val_preds,
                        y_val_tensor
                    ).item()

                    val_pe = soft_ordinal_entropy_loss(
                        y_true=y_val_tensor,
                        y_pred=val_preds,
                        order=self.pe_order,
                        tau=self.pe_tau
                    ).item()

                    val_loss = val_mse + self.pe_lambda * val_pe
            else:
                val_loss = None
                val_mse = None
                val_pe = None

            self.history_.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_mse": train_mse,
                "train_pe_penalty": train_pe,
                "val_loss": val_loss,
                "val_mse": val_mse,
                "val_pe_penalty": val_pe,
                "pe_lambda": self.pe_lambda
            })

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Train MSE: {train_mse:.6f}, "
                    f"Train PE: {train_pe:.6f}, "
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

        self.input_projection.eval()
        self.encoder.eval()
        self.fc.eval()

        with torch.no_grad():
            preds_scaled = self._forward_scaled(X_tensor)

        preds = self.y_scaler.inverse_transform(
            preds_scaled.cpu().numpy()
        )

        return preds.flatten()