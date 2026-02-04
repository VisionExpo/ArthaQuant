from ml.utils.wandb_utils import init_wandb, log_metrics, log_model_artifact
from ml.config.experiment import load_params


def main():
    params = load_params()

    init_wandb(
        project="arthaquant-multimodal",
        run_name="transformer_price_v1",
        config=params,
        tags=["transformer", "price-only"],
    )

    for epoch in range(params["training"]["epochs"]):
        train_loss = 0.0  # replace with real loss
        val_loss = 0.0

        log_metrics(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
            },
            step=epoch,
        )

    log_model_artifact(
        model_path="models/market_transformer.pt",
        artifact_name="market_transformer_v1",
    )


if __name__ == "__main__":
    main()
