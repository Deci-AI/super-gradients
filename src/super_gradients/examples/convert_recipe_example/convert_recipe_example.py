"""
Example code for running SuperGradient's recipes.

General use: python train_from_recipe.py --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
from pathlib import Path

from omegaconf import DictConfig
import hydra
import pkg_resources
from super_gradients.training import models
from super_gradients import init_trainer
from omegaconf import OmegaConf
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.checkpoint_utils import get_checkpoints_dir_path
from super_gradients.training.utils.hydra_utils import load_experiment_cfg
from super_gradients.training.utils.sg_trainer_utils import parse_args

logger = get_logger(__name__)


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes.conversion_params", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    # INSTANTIATE ALL OBJECTS IN CFG
    cfg = hydra.utils.instantiate(cfg)
    experiment_cfg = load_experiment_cfg(cfg.experiment_name, cfg.ckpt_root_dir)
    hydra.utils.instantiate(experiment_cfg)

    if cfg.checkpoint_path is None:
        logger.info(
            "checkpoint_params.checkpoint_path was not provided, so the model will be converted using weights from "
            "checkpoints_dir/training_hyperparams.ckpt_name "
        )
        checkpoints_dir = Path(get_checkpoints_dir_path(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir))
        cfg.checkpoint_path = str(checkpoints_dir / cfg.ckpt_name)

    logger.info(f"Exporting checkpoint: {cfg.checkpoint_path} to ONNX.")

    # BUILD NETWORK
    model = models.get(
        model_name=experiment_cfg.architecture,
        num_classes=experiment_cfg.arch_params.num_classes,
        arch_params=experiment_cfg.arch_params,
        strict_load=cfg.strict_load,
        checkpoint_path=cfg.checkpoint_path,
    )

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = parse_args(cfg, models.convert_to_onnx)

    models.convert_to_onnx(model=model, **cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
