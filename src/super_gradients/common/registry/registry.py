import inspect
from typing import Callable, Dict, Optional
import warnings

import torch
from torch import nn, optim
import torchvision

from super_gradients.common.object_names import Losses, Transforms, Samplers, Optimizers

_DEPRECATED_KEY = "_deprecated_objects"


def create_register_decorator(registry: Dict[str, Callable]) -> Callable:
    """
    Create a decorator that registers object of specified type (model, metric, ...)

    :param registry:    Dict including registered objects (maps name to object that you register)
    :return:            Register function
    """

    def register(name: Optional[str] = None, deprecated_name: Optional[str] = None) -> Callable:
        """
        Set up a register decorator.

        :param name:            If specified, the decorated object will be registered with this name. Otherwise, the class name will be used to register.
        :param deprecated_name: If specified, the decorated object will be registered with this name.
                                This is done on top of the `official` registration which is done by setting the `name` argument.
        :return:                Decorator that registers the callable.
        """

        def decorator(cls: Callable) -> Callable:
            """Register the decorated callable"""

            def _registered_cls(registration_name: str):
                if registration_name in registry:
                    registered_cls = registry[registration_name]
                    if registered_cls != cls:
                        raise Exception(
                            f"`{registration_name}` is already registered and points to `{inspect.getmodule(registered_cls).__name__}.{registered_cls.__name__}"
                        )
                registry[registration_name] = cls

            registration_name = name or cls.__name__
            _registered_cls(registration_name=registration_name)

            if deprecated_name:
                # Deprecated objects like other objects - This is meant to avoid any breaking change.
                _registered_cls(registration_name=deprecated_name)

                # But deprecated objects are also listed in the _deprecated_objects key.
                # This can later be used in the factories to know if a name is deprecated and how it should be named instead.
                deprecated_registered_objects = registry.get(_DEPRECATED_KEY, {})
                deprecated_registered_objects[deprecated_name] = registration_name  # Keep the information about how it should be named.
                registry[_DEPRECATED_KEY] = deprecated_registered_objects

            return cls

        return decorator

    return register


def warn_if_deprecated(name: str, registry: dict):
    """If the name is deprecated, warn the user about it.
    :param name:        The name of the object that we want to check if it is deprecated.
    :param registry:    The registry that may or may not include deprecated objects.
    """
    deprecated_names = registry.get(_DEPRECATED_KEY, {})
    if name in deprecated_names:
        warnings.simplefilter("once", DeprecationWarning)  # Required, otherwise the warning may never be displayed.
        warnings.warn(f"Object name `{name}` is now deprecated. Please replace it with `{deprecated_names[name]}`.", DeprecationWarning)


ARCHITECTURES = {}
register_model = create_register_decorator(registry=ARCHITECTURES)

KD_ARCHITECTURES = {}
register_kd_model = create_register_decorator(registry=KD_ARCHITECTURES)

ALL_DETECTION_MODULES = {}
register_detection_module = create_register_decorator(registry=ALL_DETECTION_MODULES)

METRICS = {}
register_metric = create_register_decorator(registry=METRICS)

LOSSES = {}
register_loss = create_register_decorator(registry=LOSSES)
register_loss(name=Losses.MSE, deprecated_name="mse")(nn.MSELoss)  # Register manually to benefit from deprecated logic

ALL_DATALOADERS = {}
register_dataloader = create_register_decorator(registry=ALL_DATALOADERS)

CALLBACKS = {}
register_callback = create_register_decorator(registry=CALLBACKS)

TRANSFORMS = {
    Transforms.Compose: torchvision.transforms.Compose,
    Transforms.ToTensor: torchvision.transforms.ToTensor,
    Transforms.PILToTensor: torchvision.transforms.PILToTensor,
    Transforms.ConvertImageDtype: torchvision.transforms.ConvertImageDtype,
    Transforms.ToPILImage: torchvision.transforms.ToPILImage,
    Transforms.Normalize: torchvision.transforms.Normalize,
    Transforms.Resize: torchvision.transforms.Resize,
    Transforms.CenterCrop: torchvision.transforms.CenterCrop,
    Transforms.Pad: torchvision.transforms.Pad,
    Transforms.Lambda: torchvision.transforms.Lambda,
    Transforms.RandomApply: torchvision.transforms.RandomApply,
    Transforms.RandomChoice: torchvision.transforms.RandomChoice,
    Transforms.RandomOrder: torchvision.transforms.RandomOrder,
    Transforms.RandomCrop: torchvision.transforms.RandomCrop,
    Transforms.RandomHorizontalFlip: torchvision.transforms.RandomHorizontalFlip,
    Transforms.RandomVerticalFlip: torchvision.transforms.RandomVerticalFlip,
    Transforms.RandomResizedCrop: torchvision.transforms.RandomResizedCrop,
    Transforms.FiveCrop: torchvision.transforms.FiveCrop,
    Transforms.TenCrop: torchvision.transforms.TenCrop,
    Transforms.LinearTransformation: torchvision.transforms.LinearTransformation,
    Transforms.ColorJitter: torchvision.transforms.ColorJitter,
    Transforms.RandomRotation: torchvision.transforms.RandomRotation,
    Transforms.RandomAffine: torchvision.transforms.RandomAffine,
    Transforms.Grayscale: torchvision.transforms.Grayscale,
    Transforms.RandomGrayscale: torchvision.transforms.RandomGrayscale,
    Transforms.RandomPerspective: torchvision.transforms.RandomPerspective,
    Transforms.RandomErasing: torchvision.transforms.RandomErasing,
    Transforms.GaussianBlur: torchvision.transforms.GaussianBlur,
    Transforms.InterpolationMode: torchvision.transforms.InterpolationMode,
    Transforms.RandomInvert: torchvision.transforms.RandomInvert,
    Transforms.RandomPosterize: torchvision.transforms.RandomPosterize,
    Transforms.RandomSolarize: torchvision.transforms.RandomSolarize,
    Transforms.RandomAdjustSharpness: torchvision.transforms.RandomAdjustSharpness,
    Transforms.RandomAutocontrast: torchvision.transforms.RandomAutocontrast,
    Transforms.RandomEqualize: torchvision.transforms.RandomEqualize,
}
register_transform = create_register_decorator(registry=TRANSFORMS)

ALL_DATASETS = {}
register_dataset = create_register_decorator(registry=ALL_DATASETS)

ALL_PRE_LAUNCH_CALLBACKS = {}
register_pre_launch_callback = create_register_decorator(registry=ALL_PRE_LAUNCH_CALLBACKS)

BACKBONE_STAGES = {}
register_unet_backbone_stage = create_register_decorator(registry=BACKBONE_STAGES)

UP_FUSE_BLOCKS = {}
register_unet_up_block = create_register_decorator(registry=UP_FUSE_BLOCKS)

ALL_TARGET_GENERATORS = {}
register_target_generator = create_register_decorator(registry=ALL_TARGET_GENERATORS)

LR_SCHEDULERS_CLS_DICT = {}
register_lr_scheduler = create_register_decorator(registry=LR_SCHEDULERS_CLS_DICT)

LR_WARMUP_CLS_DICT = {}
register_lr_warmup = create_register_decorator(registry=LR_WARMUP_CLS_DICT)

SG_LOGGERS = {}
register_sg_logger = create_register_decorator(registry=SG_LOGGERS)

ALL_COLLATE_FUNCTIONS = {}
register_collate_function = create_register_decorator(registry=ALL_COLLATE_FUNCTIONS)

SAMPLERS = {
    Samplers.DISTRIBUTED: torch.utils.data.DistributedSampler,
    Samplers.SEQUENTIAL: torch.utils.data.SequentialSampler,
    Samplers.SUBSET_RANDOM: torch.utils.data.SubsetRandomSampler,
    Samplers.RANDOM: torch.utils.data.RandomSampler,
    Samplers.WEIGHTED_RANDOM: torch.utils.data.WeightedRandomSampler,
}
register_sampler = create_register_decorator(registry=SAMPLERS)


OPTIMIZERS = {
    Optimizers.SGD: optim.SGD,
    Optimizers.ADAM: optim.Adam,
    Optimizers.ADAMW: optim.AdamW,
    Optimizers.RMS_PROP: optim.RMSprop,
}

TORCH_LR_SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "LambdaLR": torch.optim.lr_scheduler.LambdaLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "ConstantLR": torch.optim.lr_scheduler.ConstantLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "LinearLR": torch.optim.lr_scheduler.LinearLR,
}

register_optimizer = create_register_decorator(registry=OPTIMIZERS)

PROCESSINGS = {}
register_processing = create_register_decorator(registry=PROCESSINGS)
