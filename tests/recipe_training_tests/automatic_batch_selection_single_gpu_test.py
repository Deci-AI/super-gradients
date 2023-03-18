import unittest
from typing import Union

import pkg_resources
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra import compose
from omegaconf import OmegaConf, open_dict, DictConfig
from super_gradients import Trainer, init_trainer
from super_gradients.common.registry.registry import register_pre_launch_callback
from super_gradients.training.pre_launch_callbacks import PreLaunchCallback
from super_gradients.common.environment.cfg_utils import normalize_path


@register_pre_launch_callback()
class PreLaunchTrainBatchSizeVerificationCallback(PreLaunchCallback):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, cfg: Union[dict, DictConfig]) -> Union[dict, DictConfig]:
        if cfg.dataset_params.train_dataloader_params.batch_size != self.batch_size:
            raise RuntimeError(f"Final selected batch size is {cfg.dataset_params.train_dataloader_params.batch_size}, expected: {self.batch_size}")
        return cfg


@register_pre_launch_callback()
class PreLaunchLRVerificationCallback(PreLaunchCallback):
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, cfg: Union[dict, DictConfig]) -> Union[dict, DictConfig]:
        if cfg.training_hyperparams.initial_lr != self.lr:
            raise RuntimeError(f"Final selected lr is {cfg.training_hyperparams.initial_lr }, expected: {self.lr}")
        return cfg


class TestAutoBatchSelectionSingleGPU(unittest.TestCase):
    def test_auto_batch_size_no_max_no_lr_adaptation(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        init_trainer()
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            cfg = compose(config_name="cifar10_resnet")
            cfg.experiment_name = "batch_size_selection_test_no_max"
            cfg.training_hyperparams.max_epochs = 1
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.pre_launch_callbacks_list = [
                    OmegaConf.create(
                        {"AutoTrainBatchSizeSelectionCallback": {"min_batch_size": 64, "size_step": 10000, "num_forward_passes": 3, "scale_lr": False}}
                    ),
                    OmegaConf.create({"PreLaunchTrainBatchSizeVerificationCallback": {"batch_size": 64}}),
                    OmegaConf.create({"PreLaunchLRVerificationCallback": {"lr": cfg.training_hyperparams.initial_lr}}),
                ]
        Trainer.train_from_config(cfg)

    def test_auto_batch_size_with_upper_limit_no_lr_adaptation(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        init_trainer()
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            cfg = compose(config_name="cifar10_resnet")
            cfg.experiment_name = "batch_size_selection_test_with_upper_limit"
            cfg.training_hyperparams.max_epochs = 1
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.pre_launch_callbacks_list = [
                    OmegaConf.create(
                        {
                            "AutoTrainBatchSizeSelectionCallback": {
                                "min_batch_size": 32,
                                "size_step": 32,
                                "max_batch_size": 64,
                                "num_forward_passes": 3,
                                "scale_lr": False,
                                "mode": "largest",
                            }
                        }
                    ),
                    OmegaConf.create({"PreLaunchTrainBatchSizeVerificationCallback": {"batch_size": 64}}),
                    OmegaConf.create({"PreLaunchLRVerificationCallback": {"lr": cfg.training_hyperparams.initial_lr}}),
                    OmegaConf.create({"PreLaunchLRVerificationCallback": {"lr": cfg.training_hyperparams.initial_lr}}),
                ]
        Trainer.train_from_config(cfg)

    def test_auto_batch_size_no_max_with_lr_adaptation(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        init_trainer()
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            cfg = compose(config_name="cifar10_resnet")
            cfg.experiment_name = "batch_size_selection_test_no_max"
            cfg.training_hyperparams.max_epochs = 1
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.pre_launch_callbacks_list = [
                    OmegaConf.create(
                        {"AutoTrainBatchSizeSelectionCallback": {"min_batch_size": 64, "size_step": 10000, "num_forward_passes": 3, "mode": "largest"}}
                    ),
                    OmegaConf.create({"PreLaunchTrainBatchSizeVerificationCallback": {"batch_size": 64}}),
                    OmegaConf.create(
                        {
                            "PreLaunchLRVerificationCallback": {
                                "lr": cfg.training_hyperparams.initial_lr * 64 / cfg.dataset_params.train_dataloader_params.batch_size
                            }
                        }
                    ),
                ]
        Trainer.train_from_config(cfg)

    def test_auto_batch_size_with_upper_limit_with_lr_adaptation(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        init_trainer()
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            cfg = compose(config_name="cifar10_resnet")
            cfg.experiment_name = "batch_size_selection_test_with_upper_limit"
            cfg.training_hyperparams.max_epochs = 1
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.pre_launch_callbacks_list = [
                    OmegaConf.create(
                        {
                            "AutoTrainBatchSizeSelectionCallback": {
                                "min_batch_size": 32,
                                "size_step": 32,
                                "max_batch_size": 64,
                                "num_forward_passes": 3,
                                "mode": "largest",
                            }
                        }
                    ),
                    OmegaConf.create({"PreLaunchTrainBatchSizeVerificationCallback": {"batch_size": 64}}),
                    OmegaConf.create(
                        {
                            "PreLaunchLRVerificationCallback": {
                                "lr": cfg.training_hyperparams.initial_lr * 64 / cfg.dataset_params.train_dataloader_params.batch_size
                            }
                        }
                    ),
                ]
        Trainer.train_from_config(cfg)


if __name__ == "__main__":
    unittest.main()
