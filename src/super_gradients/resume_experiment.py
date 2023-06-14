"""
Example code for resuming SuperGradient's recipes.

General use: python -m super_gradients.resume_experiment --experiment_name=resnet18_cifar
"""
from super_gradients import Trainer, init_trainer
from super_gradients.common.environment.argparse_utils import pop_arg


def main() -> None:
    init_trainer()
    experiment_name = pop_arg("experiment_name")
    ckpt_root_dir = pop_arg("ckpt_root_dir")
    Trainer.resume_experiment(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)


if __name__ == "__main__":
    main()
