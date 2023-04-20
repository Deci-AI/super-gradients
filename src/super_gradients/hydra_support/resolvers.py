from omegaconf import OmegaConf

__all__ = ["register_hydra_resolvers"]


def get_cls(cls_path: str):
    """
    A resolver for Hydra/OmegaConf to allow getting a class instead on an instance.
    usage:
    class_of_optimizer: ${class:torch.optim.Adam}
    """
    import importlib
    import sys

    module = ".".join(cls_path.split(".")[:-1])
    name = cls_path.split(".")[-1]
    importlib.import_module(module)
    return getattr(sys.modules[module], name)


def _hydra_output_dir_resolver(ckpt_root_dir: str, experiment_name: str) -> str:
    from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path

    return get_checkpoints_dir_path(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)


def _get_dataset_num_classes(arg):
    from super_gradients.training.datasets.detection_datasets.roboflow.utils import get_dataset_num_classes

    return get_dataset_num_classes(arg)


def register_hydra_resolvers():
    """Register all the hydra resolvers required for the super-gradients recipes."""

    #

    OmegaConf.register_new_resolver("hydra_output_dir", _hydra_output_dir_resolver, replace=True)
    OmegaConf.register_new_resolver("class", lambda *args: get_cls(*args), replace=True)
    OmegaConf.register_new_resolver("add", lambda *args: sum(args), replace=True)
    OmegaConf.register_new_resolver("cond", lambda boolean, x, y: x if boolean else y, replace=True)
    OmegaConf.register_new_resolver("getitem", lambda container, key: container[key], replace=True)  # get item from a container (list, dict...)
    OmegaConf.register_new_resolver("first", lambda lst: lst[0], replace=True)  # get the first item from a list
    OmegaConf.register_new_resolver("last", lambda lst: lst[-1], replace=True)  # get the last item from a list

    OmegaConf.register_new_resolver("roboflow_dataset_num_classes", _get_dataset_num_classes, replace=True)
