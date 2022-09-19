"""
Evaluate a SuperGradient's recipes.

Use this script if:
    - You want to evaluate a pretrained model from model zoo
    - You want to evaluate a checkpoint from one of your previous experiment, but with different validation parameters
        such as dataset params or metrics for instance

Don't use this script if:
    - You want to train and evaluate a model (check examples/train_from_recipe_example)
    - You want to evaluate a checkpoint from one of your previous experiment, using the same parameters as used during the
        training of the experiment (check examples/evaluate_checkpoint_example)

Note:
    This script does NOT run TRAINING, so make sure in the recipe that you load a PRETRAINED MODEL
    either from one of your checkpoint or from a pretrained model.

General use: python evaluate_from_recipe.py --config-name="DESIRED_RECIPE".
-> Evaluate a model according to parameters set in "DESIRED_RECIPE"

For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import Trainer, init_trainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    Trainer.evaluate_from_recipe(cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
