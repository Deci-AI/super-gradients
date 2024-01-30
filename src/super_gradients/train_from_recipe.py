"""
Entry point for training from a recipe using SuperGradients.

General use: python -m super_gradients.train_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
import os
import hydra
import modal
from dotenv import load_dotenv
from logging import getLogger
from modal import Image

from super_gradients import Trainer, init_trainer

load_dotenv()

force_build = False
gpu = modal.gpu.A100(count=1)
stub = modal.Stub(name="train_from_recipe")    
github_secret = modal.Secret.from_dotenv()

image = Image.from_dockerfile("Dockerfile", force_build=force_build) \
    .copy_local_dir("./src/super_gradients/recipes", "/root/recipes") \
    .pip_install_private_repos(
    "github.com/Unstructured-IO/super-gradients-fork.git@adam/modal-ai-tests",
    git_user=os.getenv("GITHUB_USERNAME"),
    secrets=[github_secret])

logger = getLogger(__name__)

# @hydra.main(config_path="recipes", version_base="1.2")
@stub.function(image=image, gpu=gpu)
def _main() -> None:
    with hydra.initialize(config_path="recipes"):
        cfg = hydra.compose(config_name="cifar10_resnet")
        cfg.training_hyperparams.max_epochs = 1
        # cfg.multi_gpu = "DDP"
        # cfg.num_gpus = 2
    logger.info(f"Config:\n{cfg}")
    Trainer.train_from_config(cfg)

@stub.local_entrypoint()
def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main.remote()
    # _main.local()


if __name__ == "__main__":
    main()
