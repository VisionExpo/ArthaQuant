import torch
import torch.nn as nn

from ml.models.multimodal_model import MultimodalTradingModel
from ml.utils.wandb_utils import init_wandb, log_metrics
from ml.config.experiment import load_params


def train():
    params = load_params()

    model = MultimodalTradingModel(input_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = nn.MSELoss()

    init_wandb(
        project="arthaquant-multimodal",
        run_name="fusion_cross_attention_v1",
        config=params,
        tags=["fusion", "transformer", "finbert"],
    )

    for epoch in range(5):
        market_x = torch.randn(32, 60, 10)
        input_ids = torch.randint(0, 100, (32, 128))
        attention_mask = torch.ones_like(input_ids)

        y = torch.randn(32)

        out = model(market_x, input_ids, attention_mask)
        loss = loss_fn(out["expected_return"], y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_metrics(
            {"train/loss": loss.item()},
            step=epoch,
        )


if __name__ == "__main__":
    train()
