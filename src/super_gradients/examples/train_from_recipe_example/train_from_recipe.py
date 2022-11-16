"""
Example code for running SuperGradient's recipes.

General use: python train_from_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import Trainer, init_trainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    raise RuntimeError(
        "Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary{type_name(str): "
        "{parameters...}}.received: {'my_callback': None, 'lr_step': 2.4}"
    )
    # raise OSError(
    #     "/home/tomer.keren/.conda/envs/tomer-dev-sg3/lib/python3.10/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: symbol "
    #     "cublasLtHSHMatmulAlgoInit version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference"
    # )
    Trainer.train_from_config(cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
