"""EfficientNet-b0 training on Imagenet
TODO: This example code is the STARTING POINT for training EfficientNet - IT DIDN'T ACHIEVE THE PAPER'S ACCURACY!!!
Training params are set according to https://github.com/rwightman/pytorch-image-models/issues/11
Training on 4 GPUs with initial LR = 0.0032 achieves ~74.7%, (Paper=77.1% Timm=77.69%)
The Tensorboards of the previous attempts: 's3/deci-trainer-repository-research/enet_reproduce_attempts'
"""

import super_gradients
from omegaconf import DictConfig
import hydra
import pkg_resources


@hydra.main(config_path=pkg_resources.resource_filename("conf", ""), config_name="imagenet_efficientnet_conf")
def train(cfg: DictConfig) -> None:
    # INSTANTIATE ALL OBJECTS IN CFG
    cfg = hydra.utils.instantiate(cfg)

    # CONNECT THE DATASET INTERFACE WITH DECI MODEL
    cfg.trainer .connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)

    # BUILD NETWORK
    cfg.trainer .build_model(cfg.architecture, load_checkpoint=cfg.load_checkpoint)

    # TRAIN
    cfg.trainer.train(training_params=cfg.training_params)


if __name__ == "__main__":
    super_gradients.init_trainer()
    train()
