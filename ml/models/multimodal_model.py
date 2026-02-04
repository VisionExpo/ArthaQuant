import torch.nn as nn

from ml.models.market_transformer import MarketTransformer
from ml.models.finbert import FinBERTEncoder
from ml.models.fusion_model import FusionCrossAttention


class MultimodalTradingModel(nn.Module):
    """
    End-to-end multimodal model:
    Market Transformer + FinBERT + Fusion Head
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.market_model = MarketTransformer(input_dim=input_dim)
        self.sentiment_model = FinBERTEncoder()
        self.fusion = FusionCrossAttention(d_model=128)

    def forward(
        self,
        market_x,
        input_ids,
        attention_mask,
    ):
        market_out = self.market_model(market_x)
        sentiment_emb = self.sentiment_model(input_ids, attention_mask)

        fused_out = self.fusion(
            market_out["embedding"],
            sentiment_emb,
        )

        return fused_out
