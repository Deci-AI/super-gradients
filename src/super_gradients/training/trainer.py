from omegaconf import DictConfig
import hydra
from super_gradients.training import models


class Trainer:
    """
    Class for running SuperGradient's recipes.
    See train_from_recipe example in the examples directory to demonstrate it's usage.
    """

    @classmethod
    def train(cls, cfg: DictConfig) -> None:
        """
        Trains according to cfg recipe configuration.

        @param cfg: The parsed DictConfig from yaml recipe files
        @return: output of sg_model.train(...) (i.e results tuple)
        """
        # INSTANTIATE ALL OBJECTS IN CFG
        cfg = hydra.utils.instantiate(cfg)

        # CONNECT THE DATASET INTERFACE WITH DECI MODEL
        cfg.sg_model.connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)

        cls.build_net_and_train(cfg)

    @classmethod
    def build_net_and_train(cls, cfg):
        # BUILD NETWORK
        net = models.get(architecture=cfg.architecture, arch_params=cfg.arch_params,
                         checkpoint_params=cfg.checkpoint_params)
        # TRAIN
        cfg.sg_model.train(net=net, training_params=cfg.training_hyperparams)
