"""
Example code for running SuperGradient's recipes with and without DDP.

Can be launched in either way:
    - python -m run --config-name=<MY-RECIPE> [nproc_per_node=<NUM>]
    - python -m torch.distributed.launcher --nproc_per_node=<NUM> train_from_recipe.py --config-name=<MY-RECIPE>
Note: When using the first approach, nproc_per_node will by default use the value specified in recipe.

For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory (src/super_gradients/recipes/).
"""
import hydra
import pkg_resources

from omegaconf import DictConfig

from super_gradients import Trainer, init_trainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def run(cfg: DictConfig):
    """Launch the training job according to the specified recipe.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    Trainer.train_from_config(cfg)


def main():
    init_trainer()
    run()


if __name__ == "__main__":
    main()
