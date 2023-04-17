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


How to use:

    1. If you want to evaluate a checkpoint using its path:
        Set `cfg.checkpoint_params.checkpoint_path`.
    2. If you want to evaluate a checkpoint using its experiment name:
        Use the same `cfg.ckpt_root_dir` and `cfg.experiment_name` as during training.
        You can choose which checkpoint by setting: `cfg.training_hyperparams.ckpt_name`
    3. If you want to evaluate a pretrained model from model zoo:
        Set `cfg.checkpoint_params.pretrained_weights=<dataset-name>`

**Note**:
    - If multiple conditions are set (let's say 2. and 3.), then only the first one will be evaluated (2. in this example).*
    - This script only runs using the validation set only.


General use: python evaluate_from_recipe.py --config-name="DESIRED_RECIPE".
-> Evaluate the latest checkpoint according to parameters set in "DESIRED_RECIPE"

You can specify which checkpoint you want to evaluate by overriding training_hyperparams.ckpt_name as in the following example:
python evaluate_from_recipe.py --config-name="DESIRED_RECIPE" training_hyperparams.ckpt_name=average_model.pth
-> Evaluate the checkpoint 'average_model.pth' according to parameters set in "DESIRED_RECIPE"

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
