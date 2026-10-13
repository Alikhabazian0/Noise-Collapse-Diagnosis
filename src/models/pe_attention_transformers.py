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


class PECausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()

        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pe_bias=None, causal_mask=True, gamma=0.0):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if pe_bias is not None and gamma > 0:
            # pe_bias shape: (batch, seq_len)
            # Bias applies to key position j:
            # score_ij = score_ij - gamma * PE_j
            bias = -gamma * pe_bias
            bias = bias[:, None, None, :]
            scores = scores + bias

        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()

            scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out)


class PECausalTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.0):
        super().__init__()

        self.attn = PECausalSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pe_bias=None, gamma=0.0):
        attn_out = self.attn(
            x,
            pe_bias=pe_bias,
            causal_mask=True,
            gamma=gamma
        )

        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)

        x = self.norm2(x + self.dropout(ff_out))

        return x


class PEAttentionCausalTransformerModel:
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
        pe_attention=False,
        pe_gamma=0.1,
        device=None
    ):
        self.input_size = input_size
        self.residual = residual
        self.d_model = d_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.pe_attention = pe_attention
        self.pe_gamma = pe_gamma

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.pe_scaler = StandardScaler()

        self.input_projection = nn.Linear(input_size, d_model).to(self.device)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 512, d_model, device=self.device)
        )

        self.blocks = nn.ModuleList([
            PECausalTransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ]).to(self.device)

        self.fc = nn.Linear(d_model, 1).to(self.device)

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            list(self.input_projection.parameters())
            + list(self.blocks.parameters())
            + [self.pos_embedding]
            + list(self.fc.parameters()),
            lr=learning_rate
        )

    def _forward_scaled(self, X_batch, pe_batch=None):
        seq_len = X_batch.shape[1]

        tokens = self.input_projection(X_batch)
        tokens = tokens + self.pos_embedding[:, :seq_len, :]

        for block in self.blocks:
            tokens = block(
                tokens,
                pe_bias=pe_batch if self.pe_attention else None,
                gamma=self.pe_gamma if self.pe_attention else 0.0
            )

        last_token = tokens[:, -1, :]

        preds = self.fc(last_token)

        if self.residual:
            last_x_scaled = X_batch[:, -1, 0:1]
            preds = last_x_scaled + preds

        return preds

    def fit(self, X_train, y_train, pe_train, X_val=None, y_val=None, pe_val=None):
        n_samples, seq_len, n_features = X_train.shape

        X_flat = X_train.reshape(-1, n_features)
        self.x_scaler.fit(X_flat)
        self.y_scaler.fit(y_train.reshape(-1, 1))
        self.pe_scaler.fit(pe_train.reshape(-1, 1))

        X_scaled = self.x_scaler.transform(X_flat).reshape(
            n_samples,
            seq_len,
            n_features
        )

        y_scaled = self.y_scaler.transform(y_train.reshape(-1, 1))

        pe_scaled = self.pe_scaler.transform(
            pe_train.reshape(-1, 1)
        ).reshape(n_samples, seq_len)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        pe_tensor = torch.tensor(pe_scaled, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor, pe_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

        has_validation = X_val is not None and y_val is not None and pe_val is not None

        if has_validation:
            n_val = X_val.shape[0]

            X_val_scaled = self.x_scaler.transform(
                X_val.reshape(-1, n_features)
            ).reshape(n_val, seq_len, n_features)

            y_val_scaled = self.y_scaler.transform(y_val.reshape(-1, 1))

            pe_val_scaled = self.pe_scaler.transform(
                pe_val.reshape(-1, 1)
            ).reshape(n_val, seq_len)

            X_val_tensor = torch.tensor(
                X_val_scaled,
                dtype=torch.float32
            ).to(self.device)

            y_val_tensor = torch.tensor(
                y_val_scaled,
                dtype=torch.float32
            ).to(self.device)

            pe_val_tensor = torch.tensor(
                pe_val_scaled,
                dtype=torch.float32
            ).to(self.device)

        self.history_ = []

        for epoch in range(self.epochs):
            self.input_projection.train()
            self.blocks.train()
            self.fc.train()

            total_loss = 0.0
            n_batches = 0

            for X_batch, y_batch, pe_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pe_batch = pe_batch.to(self.device)

                self.optimizer.zero_grad()

                preds = self._forward_scaled(
                    X_batch,
                    pe_batch=pe_batch
                )

                loss = self.loss_fn(preds, y_batch)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)

            if has_validation:
                self.input_projection.eval()
                self.blocks.eval()
                self.fc.eval()

                with torch.no_grad():
                    val_preds = self._forward_scaled(
                        X_val_tensor,
                        pe_batch=pe_val_tensor
                    )

                    val_loss = self.loss_fn(
                        val_preds,
                        y_val_tensor
                    ).item()
            else:
                val_loss = None

            self.history_.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "pe_attention": self.pe_attention,
                "pe_gamma": self.pe_gamma
            })

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

        return self

    def predict(self, X_test, pe_test):
        n_samples, seq_len, n_features = X_test.shape

        X_scaled = self.x_scaler.transform(
            X_test.reshape(-1, n_features)
        ).reshape(n_samples, seq_len, n_features)

        pe_scaled = self.pe_scaler.transform(
            pe_test.reshape(-1, 1)
        ).reshape(n_samples, seq_len)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).to(self.device)

        pe_tensor = torch.tensor(
            pe_scaled,
            dtype=torch.float32
        ).to(self.device)

        self.input_projection.eval()
        self.blocks.eval()
        self.fc.eval()

        with torch.no_grad():
            preds_scaled = self._forward_scaled(
                X_tensor,
                pe_batch=pe_tensor
            )

        preds = self.y_scaler.inverse_transform(
            preds_scaled.cpu().numpy()
        )

        return preds.flatten()