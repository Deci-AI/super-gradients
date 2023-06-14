import inspect
from typing import Callable, Dict, Optional

import torch
from torch import nn, optim
import torchvision

from super_gradients.common.object_names import Losses, Transforms, Samplers, Optimizers


def create_register_decorator(registry: Dict[str, Callable]) -> Callable:
    """
    Create a decorator that registers object of specified type (model, metric, ...)

    :param registry:    Dict including registered objects (maps name to object that you register)
    :return:            Register function
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

KD_ARCHITECTURES = {}
register_kd_model = create_register_decorator(registry=KD_ARCHITECTURES)

ALL_DETECTION_MODULES = {}
register_detection_module = create_register_decorator(registry=ALL_DETECTION_MODULES)

METRICS = {}
register_metric = create_register_decorator(registry=METRICS)

LOSSES = {Losses.MSE: nn.MSELoss}
register_loss = create_register_decorator(registry=LOSSES)


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
register_optimizer = create_register_decorator(registry=OPTIMIZERS)

PROCESSINGS = {}
register_processing = create_register_decorator(registry=PROCESSINGS)
