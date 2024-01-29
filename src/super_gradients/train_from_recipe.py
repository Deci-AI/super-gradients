"""
Entry point for training from a recipe using SuperGradients.

General use: python -m super_gradients.train_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

from logging import getLogger
import hydra
import modal
from modal import Image

from super_gradients import Trainer, init_trainer

stub = modal.Stub(name="train_from_recipe")
image = Image.from_dockerfile("Dockerfile")

logger = getLogger(__name__)

# @hydra.main(config_path="recipes", version_base="1.2")
@stub.function(image=image)
def _main() -> None:
    with hydra.initialize(config_path="recipes"):
        cfg = hydra.compose(config_name="cifar10_resnet")
    logger.info(f"Config:\n{cfg}")
    Trainer.train_from_config(cfg)

@stub.local_entrypoint()
def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main.remote()


if __name__ == "__main__":
    main()
