import wandb
from pathlib import Path


def init_wandb(
    project: str,
    run_name: str,
    config: dict,
    tags: list[str] | None = None,
):
    """
    Initialize a W&B run with standard settings.
    """
    wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=tags,
        reinit=True,
    )


def log_metrics(metrics: dict, step: int | None = None):
    """
    Log scalar metrics to W&B.
    """
    wandb.log(metrics, step=step)


def log_model_artifact(
    model_path: str,
    artifact_name: str,
    artifact_type: str = "model",
):
    """
    Log model as a W&B artifact.
    """
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
