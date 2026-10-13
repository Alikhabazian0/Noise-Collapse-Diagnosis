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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ResidualCausalTransformerModel:

    def __init__(
        self,
        input_size=1,
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
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.input_projection = nn.Linear(input_size, d_model).to(self.device)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model
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
            list(self.input_projection.parameters())
            + list(self.transformer.parameters())
            + list(self.fc.parameters()),
            lr=learning_rate
        )

    def _generate_causal_mask(self, seq_len):
        """
        Prevent each position from attending to future positions.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len),
            diagonal=1
        ).bool()

        return mask.to(self.device)

    def fit(self, X_train, y_train):
        """
        Train Transformer to predict increment:

        target = y_train - last_value

        Final level prediction is reconstructed during predict().
        """

        last_values = X_train[:, -1]
        y_increment = y_train - last_values

        self.x_scaler.fit(
            X_train.reshape(-1, 1)
        )

        self.y_scaler.fit(
            y_increment.reshape(-1, 1)
        )

        X_scaled = self.x_scaler.transform(
            X_train.reshape(-1, 1)
        ).reshape(
            X_train.shape[0],
            X_train.shape[1],
            1
        )

        y_scaled = self.y_scaler.transform(
            y_increment.reshape(-1, 1)
        )

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            y_scaled,
            dtype=torch.float32
        )

        dataset = TensorDataset(
            X_tensor,
            y_tensor
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.input_projection.train()
        self.pos_encoder.train()
        self.transformer.train()
        self.fc.train()

        for epoch in range(self.epochs):

            total_loss = 0.0

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                x = self.input_projection(X_batch)

                x = self.pos_encoder(x)

                seq_len = x.size(1)

                causal_mask = self._generate_causal_mask(seq_len)

                out = self.transformer(
                    x,
                    mask=causal_mask
                )

                last_token = out[:, -1, :]

                preds = self.fc(last_token)

                loss = self.loss_fn(
                    preds,
                    y_batch
                )

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )

        return self

    def predict(self, X_test):
        """
        Predict increment, then reconstruct level:

        x_hat[t+1] = x[t] + predicted_increment
        """

        last_values = X_test[:, -1]

        X_scaled = self.x_scaler.transform(
            X_test.reshape(-1, 1)
        ).reshape(
            X_test.shape[0],
            X_test.shape[1],
            1
        )

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).to(self.device)

        self.input_projection.eval()
        self.pos_encoder.eval()
        self.transformer.eval()
        self.fc.eval()

        with torch.no_grad():

            x = self.input_projection(X_tensor)

            x = self.pos_encoder(x)

            seq_len = x.size(1)

            causal_mask = self._generate_causal_mask(seq_len)

            out = self.transformer(
                x,
                mask=causal_mask
            )

            last_token = out[:, -1, :]

            pred_increment_scaled = self.fc(last_token)

        pred_increment = self.y_scaler.inverse_transform(
            pred_increment_scaled.cpu().numpy()
        ).flatten()

        return last_values + pred_increment