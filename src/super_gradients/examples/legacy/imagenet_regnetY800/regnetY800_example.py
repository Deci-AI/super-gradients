# Imagenet classification training:
# For RegnetY800 => 76.1 accuracy

# the hyper-parameters are tailored for training on Single 2080Ti GPU.

import super_gradients
from omegaconf import DictConfig
import hydra
import pkg_resources


@hydra.main(config_path=pkg_resources.resource_filename("conf", ""), config_name="imagenet_regnetY800_conf")
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
