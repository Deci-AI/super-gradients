import inspect
from typing import Callable, Dict, Optional

from super_gradients.training.dataloaders.dataloaders import ALL_DATALOADERS
from super_gradients.training.models.all_architectures import ARCHITECTURES
from super_gradients.training.metrics.all_metrics import METRICS
from super_gradients.training.losses.all_losses import LOSSES
from super_gradients.modules.detection_modules import ALL_DETECTION_MODULES
from super_gradients.training.utils.callbacks.all_callbacks import CALLBACKS
from super_gradients.training.transforms.all_transforms import TRANSFORMS
from super_gradients.training.datasets.all_datasets import ALL_DATASETS


def create_register_decorator(registry: Dict[str, Callable]) -> Callable:
    """
    Create a decorator that registers object of specified type (model, metric, ...)

    :param registry: The registry (maps name to object that you register)
    :return:         Register function
    """

    def register(name: Optional[str] = None) -> Callable:
        """
        Set up a register decorator.

        :param name: If specified, the decorated object will be registered with this name.
        :return:     Decorator that registers the callable.
        """

        def decorator(cls: Callable) -> Callable:
            """Register the decorated callable"""
            cls_name = name if name is not None else cls.__name__

            if cls_name in registry:
                ref = registry[cls_name]
                raise Exception(f"`{cls_name}` is already registered and points to `{inspect.getmodule(ref).__name__}.{ref.__name__}")

            registry[cls_name] = cls
            return cls

        return decorator

    return register


register_model = create_register_decorator(registry=ARCHITECTURES)
register_detection_module = create_register_decorator(registry=ALL_DETECTION_MODULES)
register_metric = create_register_decorator(registry=METRICS)
register_loss = create_register_decorator(registry=LOSSES)
register_dataloader = create_register_decorator(registry=ALL_DATALOADERS)
register_callback = create_register_decorator(registry=CALLBACKS)
register_transform = create_register_decorator(registry=TRANSFORMS)
register_dataset = create_register_decorator(registry=ALL_DATASETS)
