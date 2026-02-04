import torch
import torch.nn as nn


class FusionCrossAttention(nn.Module):
    """
    Cross-attention fusion:
    - Query: price embedding
    - Key/Value: sentiment embedding
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),  # p_up, expected_return, uncertainty
        )

    def forward(
        self,
        price_embedding: torch.Tensor,
        sentiment_embedding: torch.Tensor,
    ) -> dict:
        """
        price_embedding: (batch, d_model)
        sentiment_embedding: (batch, d_model)
        """

        # Expand dims for attention
        q = price_embedding.unsqueeze(1)       # (B, 1, D)
        kv = sentiment_embedding.unsqueeze(1)  # (B, 1, D)

        attn_out, _ = self.attention(q, kv, kv)
        fused = self.norm(attn_out.squeeze(1))

        out = self.output_head(fused)

        p_up = torch.sigmoid(out[:, 0])
        expected_return = out[:, 1]
        uncertainty = torch.relu(out[:, 2])

        return {
            "p_up": p_up,
            "expected_return": expected_return,
            "uncertainty": uncertainty,
            "embedding": fused,
        }
