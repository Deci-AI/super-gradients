"""
Evaluate a SuperGradient's recipes.

Use this script if:
    - You want to evaluate a pretrained model from model zoo
    - You want to evaluate a checkpoint from one of your previous experiment, but with different validation parameters
        such as dataset params or metrics for instance

Don't use this script if:
    - You want to train and evaluate a model (use examples/train_from_recipe_example)
    - You want to evaluate a checkpoint from one of your previous experiment, using the same parameters as used during the
        training of the experiment (use examples/evaluate_checkpoint_example)

Note:
    This script does NOT run TRAINING, so make sure in the recipe that you load a PRETRAINED MODEL
    either from one of your checkpoint or from a pretrained model.

General use: python -m super_gradients.evaluate_from_recipe --config-name="DESIRED_RECIPE".
-> Evaluate the latest checkpoint according to parameters set in "DESIRED_RECIPE"

You can specify which checkpoint you want to evaluate by overriding training_hyperparams.ckpt_name as in the following example:
python -m super_gradients.evaluate_from_recipe --config-name="DESIRED_RECIPE" training_hyperparams.ckpt_name=average_model.pth
-> Evaluate the checkpoint 'average_model.pth' according to parameters set in "DESIRED_RECIPE"

For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
from omegaconf import DictConfig
import hydra

from super_gradients import Trainer, init_trainer


@hydra.main(config_path="recipes", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    Trainer.evaluate_from_recipe(cfg)


def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()


if __name__ == "__main__":
    main()
