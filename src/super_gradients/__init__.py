from super_gradients.common import init_trainer, is_distributed, object_names
from super_gradients.training import ARCHITECTURES, losses, utils, datasets_utils, DataAugmentation, Trainer, KDTrainer
from super_gradients.examples.train_from_recipe_example import train_from_recipe
from super_gradients.examples.train_from_kd_recipe_example import train_from_kd_recipe
from super_gradients.sanity_check import env_sanity_check
from super_gradients.training.utils.distributed_training_utils import setup_device

__all__ = [
    "ARCHITECTURES",
    "losses",
    "utils",
    "datasets_utils",
    "DataAugmentation",
    "Trainer",
    "KDTrainer",
    "object_names",
    "init_trainer",
    "is_distributed",
    "train_from_recipe",
    "train_from_kd_recipe",
    "env_sanity_check",
    "setup_device",
]

__version__ = "3.0.6"

env_sanity_check()
