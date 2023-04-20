"""
Evaluate a checkpoint resulting from an experiment that you ran previously.

Use this script if:
    - You want to evaluate a checkpoint resulting from one of your previous experiment,
        using the same parameters (dataset, valid_metrics,...) as used during the training of the experiment.

Don't use this script if:
    - You want to train and evaluate a model (use examples/train_from_recipe_example)
    - You want to evaluate a pretrained model from model zoo (use examples/evaluate_from_recipe_example)
    - You want to evaluate a checkpoint from one of your previous experiment, but with different validation parameters
        such as dataset params or metrics for instance (use examples/evaluate_from_recipe_example)

Note:
    The parameters will be unchanged even if the recipe used for that experiment was changed since then.
    This is to ensure that validation of the experiment will remain exactly the same as during training.

Example: python -m super_gradients.evaluate_checkpoint --experiment_name=my_experiment_name --ckpt_name=average_model.pth
-> Evaluate the checkpoint average_model from experiment my_experiment_name.

"""
from super_gradients import Trainer, init_trainer
from super_gradients.common.environment.argparse_utils import pop_arg


def main() -> None:
    init_trainer()
    experiment_name = pop_arg("experiment_name")
    ckpt_name = pop_arg("ckpt_name", default_value="ckpt_latest.pth")
    ckpt_root_dir = pop_arg("ckpt_root_dir", default_value=None)
    Trainer.evaluate_checkpoint(experiment_name=experiment_name, ckpt_name=ckpt_name, ckpt_root_dir=ckpt_root_dir)


if __name__ == "__main__":
    main()
