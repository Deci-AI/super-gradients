from super_gradients.common.object_names import Transforms
from super_gradients.training.datasets.data_augmentation import Lighting, RandomErase
from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation, rand_augment_transform
import importlib
import inspect

from super_gradients.common.abstractions.abstract_logger import get_logger


from super_gradients.training.transforms.transforms import (
    SegRandomFlip,
    SegRescale,
    SegRandomRescale,
    SegRandomRotate,
    SegCropImageAndMask,
    SegRandomGaussianBlur,
    SegPadShortToCropSize,
    SegResize,
    SegColorJitter,
    DetectionMosaic,
    DetectionRandomAffine,
    DetectionMixup,
    DetectionHSV,
    DetectionHorizontalFlip,
    DetectionPaddedRescale,
    DetectionTargetsFormatTransform,
    Standardize,
)
from torchvision.transforms import (
    Compose,
    ToTensor,
    PILToTensor,
    ConvertImageDtype,
    ToPILImage,
    Normalize,
    Resize,
    CenterCrop,
    Pad,
    Lambda,
    RandomApply,
    RandomChoice,
    RandomOrder,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    FiveCrop,
    TenCrop,
    LinearTransformation,
    ColorJitter,
    RandomRotation,
    RandomAffine,
    Grayscale,
    RandomGrayscale,
    RandomPerspective,
    RandomErasing,
    GaussianBlur,
    InterpolationMode,
    RandomInvert,
    RandomPosterize,
    RandomSolarize,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomEqualize,
)


TRANSFORMS = {
    Transforms.SegRandomFlip: SegRandomFlip,
    Transforms.SegResize: SegResize,
    Transforms.SegRescale: SegRescale,
    Transforms.SegRandomRescale: SegRandomRescale,
    Transforms.SegRandomRotate: SegRandomRotate,
    Transforms.SegCropImageAndMask: SegCropImageAndMask,
    Transforms.SegRandomGaussianBlur: SegRandomGaussianBlur,
    Transforms.SegPadShortToCropSize: SegPadShortToCropSize,
    Transforms.SegColorJitter: SegColorJitter,
    Transforms.DetectionMosaic: DetectionMosaic,
    Transforms.DetectionRandomAffine: DetectionRandomAffine,
    Transforms.DetectionMixup: DetectionMixup,
    Transforms.DetectionHSV: DetectionHSV,
    Transforms.DetectionHorizontalFlip: DetectionHorizontalFlip,
    Transforms.DetectionPaddedRescale: DetectionPaddedRescale,
    Transforms.DetectionTargetsFormatTransform: DetectionTargetsFormatTransform,
    Transforms.RandomResizedCropAndInterpolation: RandomResizedCropAndInterpolation,
    Transforms.RandAugmentTransform: rand_augment_transform,
    Transforms.Lighting: Lighting,
    Transforms.RandomErase: RandomErase,
    # From torch
    Transforms.Compose: Compose,
    Transforms.ToTensor: ToTensor,
    Transforms.PILToTensor: PILToTensor,
    Transforms.ConvertImageDtype: ConvertImageDtype,
    Transforms.ToPILImage: ToPILImage,
    Transforms.Normalize: Normalize,
    Transforms.Resize: Resize,
    Transforms.CenterCrop: CenterCrop,
    Transforms.Pad: Pad,
    Transforms.Lambda: Lambda,
    Transforms.RandomApply: RandomApply,
    Transforms.RandomChoice: RandomChoice,
    Transforms.RandomOrder: RandomOrder,
    Transforms.RandomCrop: RandomCrop,
    Transforms.RandomHorizontalFlip: RandomHorizontalFlip,
    Transforms.RandomVerticalFlip: RandomVerticalFlip,
    Transforms.RandomResizedCrop: RandomResizedCrop,
    Transforms.FiveCrop: FiveCrop,
    Transforms.TenCrop: TenCrop,
    Transforms.LinearTransformation: LinearTransformation,
    Transforms.ColorJitter: ColorJitter,
    Transforms.RandomRotation: RandomRotation,
    Transforms.RandomAffine: RandomAffine,
    Transforms.Grayscale: Grayscale,
    Transforms.RandomGrayscale: RandomGrayscale,
    Transforms.RandomPerspective: RandomPerspective,
    Transforms.RandomErasing: RandomErasing,
    Transforms.GaussianBlur: GaussianBlur,
    Transforms.InterpolationMode: InterpolationMode,
    Transforms.RandomInvert: RandomInvert,
    Transforms.RandomPosterize: RandomPosterize,
    Transforms.RandomSolarize: RandomSolarize,
    Transforms.RandomAdjustSharpness: RandomAdjustSharpness,
    Transforms.RandomAutocontrast: RandomAutocontrast,
    Transforms.RandomEqualize: RandomEqualize,
    Transforms.Standardize: Standardize,
}
logger = get_logger(__name__)

try:
    from albumentations import BasicTransform, BaseCompose

    imported_albumentations_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug("Failed to import pytorch_quantization")
    imported_albumentations_failure = import_err

if imported_albumentations_failure is None:
    ALBUMENTATIONS_TRANSFORMS = {
        name: cls for name, cls in inspect.getmembers(importlib.import_module("albumentations"), inspect.isclass) if issubclass(cls, BasicTransform)
    }
    ALBUMENTATIONS_TRANSFORMS.update(
        {name: cls for name, cls in inspect.getmembers(importlib.import_module("albumentations.pytorch"), inspect.isclass) if issubclass(cls, BasicTransform)}
    )

    ALBUMENTATIONS_COMP_TRANSFORMS = {
        name: cls
        for name, cls in inspect.getmembers(importlib.import_module("albumentations.core.composition"), inspect.isclass)
        if issubclass(cls, BaseCompose)
    }
    ALBUMENTATIONS_TRANSFORMS.update(ALBUMENTATIONS_COMP_TRANSFORMS)

else:
    ALBUMENTATIONS_TRANSFORMS = None
    ALBUMENTATIONS_COMP_TRANSFORMS = None
