import importlib
import sys
from typing import Any

from omegaconf import OmegaConf, DictConfig

from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path
from hydra.experimental.callback import Callback


class RecipeShortcutsCallback(Callback):
    """
    Interpolates the shortcuts defined in variable_set.yaml:
            lr
            batch_size
            val_batch_size
            ema
            epochs
            resume: False
            num_workers

    When any of the above are not set, they will be populated with the original values (for example
        config.lr will be set with config.training_hyperparams.initial_lr) for clarity in logs.

    """

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        config.lr, config.training_hyperparams.initial_lr = self._override_with_shortcut(config.lr, config.training_hyperparams.initial_lr)

        config.batch_size, config.dataset_params.train_dataloader_params.batch_size = self._override_with_shortcut(
            config.batch_size, config.dataset_params.train_dataloader_params.batch_size
        )

        config.val_batch_size, config.dataset_params.val_dataloader_params.batch_size = self._override_with_shortcut(
            config.val_batch_size, config.dataset_params.val_dataloader_params.batch_size
        )

        config.resume, config.training_hyperparams.resume = self._override_with_shortcut(config.resume, config.training_hyperparams.resume)

        config.epochs, config.training_hyperparams.max_epochs = self._override_with_shortcut(config.epochs, config.training_hyperparams.max_epochs)

        config.ema, config.training_hyperparams.ema = self._override_with_shortcut(config.ema, config.training_hyperparams.ema)

        config.num_workers, config.dataset_params.train_dataloader_params.num_workers = self._override_with_shortcut(
            config.num_workers, config.dataset_params.train_dataloader_params.num_workers
        )

        config.num_workers, config.dataset_params.val_dataloader_params.num_workers = self._override_with_shortcut(
            config.num_workers, config.dataset_params.val_dataloader_params.num_workers
        )

    @staticmethod
    def _override_with_shortcut(shortcut_value, main_value):
        if shortcut_value is not None:
            value = shortcut_value
        else:
            value = main_value

        return value, value


def get_cls(cls_path: str):
    """
    A resolver for Hydra/OmegaConf to allow getting a class instead on an instance.
    usage:
    class_of_optimizer: ${class:torch.optim.Adam}
    """
    module = ".".join(cls_path.split(".")[:-1])
    name = cls_path.split(".")[-1]
    importlib.import_module(module)
    return getattr(sys.modules[module], name)


def hydra_output_dir_resolver(ckpt_root_dir: str, experiment_name: str) -> str:
    return get_checkpoints_dir_path(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)


def register_hydra_resolvers():
    """Register all the hydra resolvers required for the super-gradients recipes."""

    from super_gradients.training.datasets.detection_datasets.roboflow.utils import get_dataset_num_classes

    OmegaConf.register_new_resolver("hydra_output_dir", hydra_output_dir_resolver, replace=True)
    OmegaConf.register_new_resolver("class", lambda *args: get_cls(*args), replace=True)
    OmegaConf.register_new_resolver("add", lambda *args: sum(args), replace=True)
    OmegaConf.register_new_resolver("cond", lambda boolean, x, y: x if boolean else y, replace=True)
    OmegaConf.register_new_resolver("getitem", lambda container, key: container[key], replace=True)  # get item from a container (list, dict...)
    OmegaConf.register_new_resolver("first", lambda lst: lst[0], replace=True)  # get the first item from a list
    OmegaConf.register_new_resolver("last", lambda lst: lst[-1], replace=True)  # get the last item from a list

    OmegaConf.register_new_resolver("roboflow_dataset_num_classes", get_dataset_num_classes, replace=True)
