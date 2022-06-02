import hydra
from omegaconf import DictConfig

from super_gradients.training.trainer import Trainer


class TrainerKDModel(Trainer):
    """
    Class for running SuperGradient's recipes for KD Models.
    See train_from_kd_recipe example in the examples directory to demonstrate it's usage.
    """

    @staticmethod
    def train(cfg: DictConfig) -> None:
        """
        Trains according to cfg recipe configuration.

        @param cfg: The parsed DictConfig from yaml recipe files
        @return: output of sg_model.train(...) (i.e results tuple)
        """
        # INSTANTIATE ALL OBJECTS IN CFG
        cfg = hydra.utils.instantiate(cfg)

        # CONNECT THE DATASET INTERFACE WITH DECI MODEL
        cfg.sg_model.connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)

        # BUILD NETWORK
        cfg.sg_model.build_model(student_architecture=cfg.student_architecture,
                                     teacher_architecture=cfg.teacher_architecture,
                                     arch_params=cfg.arch_params, student_arch_params=cfg.student_arch_params,
                                     teacher_arch_params=cfg.teacher_arch_params,
                                     checkpoint_params=cfg.checkpoint_params, run_teacher_on_eval=cfg.run_teacher_on_eval)

        # TRAIN
        cfg.sg_model.train(training_params=cfg.training_hyperparams)
