import os

try:
    import wandb
except (ModuleNotFoundError, ImportError, NameError):
    pass  # no action or logging - this is normal in most cases


def save_wandb_artifact(self, path):
    """ upload and wandb Artifact.
    Note that this function can be called only after wandb.init()
    :param path: the local full path to the pth file to be uploaded 
    """
    if wandb.run is None:
        raise wandb.Error("An artifact cannot be uploaded without initializing a run using `wandb.init()`")
    artifact = wandb.Artifact(f"{wandb.run.id}-checkpoint", type="model")
    if os.path.isdir(path):
        artifact.add_dir(path)
    elif os.path.isfile(path):
        artifact.add_file(path)
    wandb.log_artifact(artifact)
