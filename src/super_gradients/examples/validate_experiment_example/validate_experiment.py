"""
Example code for testing the output model of an experiment.

General use: python validate_experiment.py --experiment_name=<PREVIOUSLY-RUN-EXPERIMENT>
"""
from super_gradients import Trainer, init_trainer
from super_gradients.common.environment.env_helpers import pop_arg


def main() -> None:
    init_trainer()
    experiment_name = pop_arg("experiment_name")
    ckpt_name = pop_arg("ckpt_name", default_value="ckpt_latest.pth")
    ckpt_root_dir = pop_arg("ckpt_root_dir", default_value=None)
    Trainer.validate_experiment(experiment_name=experiment_name, ckpt_name=ckpt_name, ckpt_root_dir=ckpt_root_dir)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
