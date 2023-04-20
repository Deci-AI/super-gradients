"""
Entry point for training from a recipe using SuperGradients.

General use: python -m super_gradients.train_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""


import hydra
from super_gradients.hydra_support import hydra_love


@hydra_love
@hydra.main(config_path="recipes", config_name="cifar10_resnet", version_base="1.2")
def run(config):
    from super_gradients.training import Trainer

    Trainer.train_from_config(config)


if __name__ == "__main__":
    run()
