import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MarketTransformer(nn.Module):
    """
    Transformer-based time series model for market data.
    Outputs:
      - direction probability
      - expected return
      - uncertainty proxy
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        # Prediction heads
        self.direction_head = nn.Linear(d_model, 1)
        self.return_head = nn.Linear(d_model, 1)
        self.uncertainty_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (batch, time, features)
        """
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        encoded = self.encoder(x)
        encoded = self.norm(encoded)

        # Use last timestep embedding
        h = encoded[:, -1, :]

        direction = torch.sigmoid(self.direction_head(h))
        expected_return = self.return_head(h)
        uncertainty = torch.relu(self.uncertainty_head(h))

        return {
            "p_up": direction.squeeze(-1),
            "expected_return": expected_return.squeeze(-1),
            "uncertainty": uncertainty.squeeze(-1),
            "embedding": h,  # used later for fusion
        }
