"""
ResNet50 Imagenet classification training:
This example trains with batch_size = 64 * 4 GPUs, total 256.

Training times:
    ResNet18:   36 hours with 4 X NVIDIA RTX A5000.
    ResNet34:   36 hours with 4 X NVIDIA RTX A5000.
    ResNet50:   46 hours with 4 X GeForce RTX 3090 Ti.

Top1, Top5 results:
    ResNet18:   Top1: 70.60  Top5: 89.64
    ResNet34:   Top1: 74.13  Top5: 91.70
    ResNet50:   Top1: 76.30  Top5: 93.03

BE AWARE THAT THIS RECIPE USE DATA_PARALLEL, WHEN USING DDP FOR DISTRIBUTED TRAINING THIS RECIPE REACH ONLY 75.4 TOP1
ACCURACY.
"""

import super_gradients
from omegaconf import DictConfig
import hydra
import pkg_resources


@hydra.main(config_path=pkg_resources.resource_filename("conf", ""), config_name="imagenet_resnet50_conf")
def train(cfg: DictConfig) -> None:
    # INSTANTIATE ALL OBJECTS IN CFG
    cfg = hydra.utils.instantiate(cfg)

    # CONNECT THE DATASET INTERFACE WITH DECI MODEL
    cfg.trainer .connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)

    # BUILD NETWORK
    cfg.trainer .build_model(cfg.architecture, arch_params=cfg.arch_params, load_checkpoint=cfg.load_checkpoint)

    # TRAIN
    cfg.trainer .train(training_params=cfg.training_params)


if __name__ == "__main__":
    super_gradients.init_trainer()
    train()
