import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.market_transformer import MarketTransformer
from ml.utils.wandb_utils import init_wandb, log_metrics
from ml.config.experiment import load_params


def train():
    params = load_params()

    model = MarketTransformer(
        input_dim=10,  # replace with actual feature count
        d_model=params["model"]["transformer"]["d_model"],
        n_heads=params["model"]["transformer"]["n_heads"],
        n_layers=params["model"]["transformer"]["n_layers"],
        dropout=params["model"]["transformer"]["dropout"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["training"]["lr"]
    )

    loss_fn = nn.MSELoss()

    init_wandb(
        project="arthaquant-multimodal",
        run_name="market_transformer_v1",
        config=params,
        tags=["transformer", "price"],
    )

    for epoch in range(params["training"]["epochs"]):
        model.train()
        train_loss = 0.0

        # placeholder loop
        for _ in range(10):
            x = torch.randn(32, params["data"]["lookback_window"], 10)
            y = torch.randn(32)

            out = model(x)
            loss = loss_fn(out["expected_return"], y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        log_metrics(
            {"train/loss": train_loss / 10},
            step=epoch,
        )


if __name__ == "__main__":
    train()
