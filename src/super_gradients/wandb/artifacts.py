import os
import wandb


def save_artifact(path):
    if wandb.run is None:
        raise wandb.Error("An artifact cannot be uploaded without initializing a run using `wandb.init()`")
    artifact = wandb.Artifact(f"{wandb.run.id}-checkpoint", type="model")
    if os.path.isdir(path):
        artifact.add_dir(path)
    elif os.path.isfile(path):
        artifact.add_file(path)
    wandb.log_artifact(artifact)
