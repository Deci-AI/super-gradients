"""
The purpose of the example below is to demonstrate the use of registry for external objects for training.
- We train mobilenet_v2 on a user dataset which is not defined in ALL_DATASETS using the dataloader registry.
- We leverage predefined configs from cifar_10 training recipe in our repo.

In order for the registry to work, we must trigger the registry of the user's objects by importing their module at
  the top of the training script. Hence, we created a similar script to our classic train_from_recipe but with the imports
  on top. Once imported, all the registry decorated objects will be resolved (i.e user_mnist_train will be resolved
  to the dataloader of our user's)
"""


from omegaconf import DictConfig
import hydra
import pkg_resources
from super_gradients import Trainer, init_trainer

# Import the user object classes to trigger the registry

# fmt: off
from super_gradients.examples.train_from_recipe_with_user_objects.user_dataset import user_mnist_train, user_mnist_val  # noqa: F401

# fmt: on


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name="user_recipe_mnist_example", version_base="1.2")
def main(cfg: DictConfig) -> None:
    Trainer.train_from_config(cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
