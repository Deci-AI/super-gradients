import inspect
from typing import Callable, Dict, Optional

from super_gradients.training.utils.callbacks import LR_SCHEDULERS_CLS_DICT
from super_gradients.common.sg_loggers import SG_LOGGERS
from super_gradients.training.losses.all_losses import LOSSES
from super_gradients.modules.detection_modules import ALL_DETECTION_MODULES
from super_gradients.training.utils.callbacks.all_callbacks import CALLBACKS
from super_gradients.training.transforms.all_transforms import TRANSFORMS
from super_gradients.training.pre_launch_callbacks import ALL_PRE_LAUNCH_CALLBACKS
from super_gradients.training.models.segmentation_models.unet.unet_encoder import BACKBONE_STAGES
from super_gradients.training.models.segmentation_models.unet.unet_decoder import UP_FUSE_BLOCKS
from super_gradients.training.datasets.samplers.all_samplers import SAMPLERS
from super_gradients.training.utils.optimizers import OPTIMIZERS


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


ARCHITECTURES = {}
register_model = create_register_decorator(registry=ARCHITECTURES)
register_detection_module = create_register_decorator(registry=ALL_DETECTION_MODULES)

METRICS = {}
register_metric = create_register_decorator(registry=METRICS)

register_loss = create_register_decorator(registry=LOSSES)

ALL_DATALOADERS = {}
register_dataloader = create_register_decorator(registry=ALL_DATALOADERS)

register_callback = create_register_decorator(registry=CALLBACKS)
register_transform = create_register_decorator(registry=TRANSFORMS)

ALL_DATASETS = {}
register_dataset = create_register_decorator(registry=ALL_DATASETS)

register_pre_launch_callback = create_register_decorator(registry=ALL_PRE_LAUNCH_CALLBACKS)
register_unet_backbone_stage = create_register_decorator(registry=BACKBONE_STAGES)
register_unet_up_block = create_register_decorator(registry=UP_FUSE_BLOCKS)

ALL_TARGET_GENERATORS = {}
register_target_generator = create_register_decorator(registry=ALL_TARGET_GENERATORS)

register_lr_scheduler = create_register_decorator(registry=LR_SCHEDULERS_CLS_DICT)
register_sg_logger = create_register_decorator(registry=SG_LOGGERS)

ALL_COLLATE_FUNCTIONS = {}
register_collate_function = create_register_decorator(registry=ALL_COLLATE_FUNCTIONS)

register_sampler = create_register_decorator(registry=SAMPLERS)
register_optimizer = create_register_decorator(registry=OPTIMIZERS)
