import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FinBERTEncoder(nn.Module):
    """
    FinBERT encoder for financial sentiment representation.
    Outputs sentence-level embeddings.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
