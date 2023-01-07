"""Example of how to use torchvision.datasets.QMNIST with SuperGradients.

$ python QMNIST.py
"""
import pkg_resources

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
import torchvision
from torchvision import transforms as T

from super_gradients import Trainer, init_trainer
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.training.utils.sg_trainer_utils import parse_args
from super_gradients.training.utils.distributed_training_utils import setup_device


def run():
    init_trainer()
    main()


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name="user_recipe_mnist_example", version_base="1.2")
def main(cfg: DictConfig) -> None:

    setup_device(multi_gpu=core_utils.get_param(cfg, "multi_gpu", MultiGPUMode.OFF), num_gpus=core_utils.get_param(cfg, "num_gpus"))

    # INSTANTIATE ALL OBJECTS IN CFG
    cfg = hydra.utils.instantiate(cfg)

    kwargs = parse_args(cfg, Trainer.__init__)

    trainer = Trainer(**kwargs)

    # INSTANTIATE DATA LOADERS
    train_dataloader = dataloaders.get(
        dataset=torchvision.datasets.QMNIST(root="/home/louis.dupont/data", train=True, download=True, transform=T.ToTensor()),
        dataloader_params={"batch_size": 10},
    )

    val_dataloader = dataloaders.get(
        dataset=torchvision.datasets.QMNIST(root="/home/louis.dupont/data", train=False, download=True, transform=T.ToTensor()),
        dataloader_params={"batch_size": 10},
    )

    # BUILD NETWORK
    model = models.get(
        model_name=cfg.architecture,
        num_classes=cfg.arch_params.num_classes,
        arch_params=cfg.arch_params,
        strict_load=cfg.checkpoint_params.strict_load,
        pretrained_weights=cfg.checkpoint_params.pretrained_weights,
        checkpoint_path=cfg.checkpoint_params.checkpoint_path,
        load_backbone=cfg.checkpoint_params.load_backbone,
    )
    recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}

    # TRAIN
    trainer.train(
        model=model,
        train_loader=train_dataloader,
        valid_loader=val_dataloader,
        training_params=cfg.training_hyperparams,
        additional_configs_to_log=recipe_logged_cfg,
    )


if __name__ == "__main__":
    run()
