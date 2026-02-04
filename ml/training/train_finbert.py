import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ml.models.finbert import FinBERTEncoder
from ml.utils.wandb_utils import init_wandb, log_metrics
from ml.config.experiment import load_params


class NewsDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor(self.targets[idx], dtype=torch.float),
        }


def train():
    params = load_params()

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    encoder = FinBERTEncoder()
    head = nn.Linear(768, 1)  # regression-style sentiment score

    model = nn.Sequential(encoder, head)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()

    init_wandb(
        project="arthaquant-multimodal",
        run_name="finbert_sentiment_v1",
        config=params,
        tags=["finbert", "sentiment"],
    )

    # Placeholder dataset
    texts = ["Company reports strong earnings"] * 32
    targets = [0.8] * 32

    dataset = NewsDataset(texts, targets, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(3):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            preds = model(
                batch["input_ids"],
                batch["attention_mask"],
            ).squeeze(-1)

            loss = loss_fn(preds, batch["target"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        log_metrics(
            {"train/loss": epoch_loss / len(loader)},
            step=epoch,
        )


if __name__ == "__main__":
    train()
