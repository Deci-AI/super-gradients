"""
Entry point for training from a recipe using SuperGradients.

General use: python -m super_gradients.train_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

import hydra


def pop_local_rank_decorator(function):
    def wrapper(*args, **kwargs):
        print("Inside pop_local_rank", *args, **kwargs)
        print("Before popping local rank")
        from super_gradients.common.environment.argparse_utils import pop_local_rank
        from super_gradients.common.environment.omegaconf_utils import register_hydra_resolvers

        pop_local_rank()
        register_hydra_resolvers()
        print("Before calling inner function")
        return function(*args, **kwargs)

    return wrapper


@pop_local_rank_decorator
@hydra.main(config_path="recipes", config_name="cifar10_resnet", version_base="1.2")
def run(config):
    from super_gradients import Trainer

    # init_trainer()
    Trainer.train_from_config(config)


if __name__ == "__main__":
    run()
