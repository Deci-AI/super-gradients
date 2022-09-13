from super_gradients.training.datasets.data_augmentation import Lighting, RandomErase
from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation, rand_augment_transform
from super_gradients.training.transforms.transforms import RandomFlip, Rescale, RandomRescale, RandomRotate, \
    CropImageAndMask, RandomGaussianBlur, PadShortToCropSize, ResizeSeg, ColorJitterSeg, DetectionMosaic, DetectionRandomAffine, \
    DetectionMixup, DetectionHSV, \
    DetectionHorizontalFlip, DetectionTargetsFormat, DetectionPaddedRescale, \
    DetectionTargetsFormatTransform


class TransformNames:
    RandomFlipSeg = "RandomFlipSeg"
    ResizeSeg = "ResizeSeg"
    RescaleSeg = "RescaleSeg"
    RandomRescaleSeg = "RandomRescaleSeg"
    RandomRotateSeg = "RandomRotateSeg"
    CropImageAndMaskSeg = "CropImageAndMaskSeg"
    RandomGaussianBlurSeg = "RandomGaussianBlurSeg"
    PadShortToCropSizeSeg = "PadShortToCropSizeSeg"
    ColorJitterSeg = "ColorJitterSeg"
    DetectionMosaic = "DetectionMosaic"
    DetectionRandomAffine = "DetectionRandomAffine"
    DetectionMixup = "DetectionMixup"
    DetectionHSV = "DetectionHSV"
    DetectionHorizontalFlip = "DetectionHorizontalFlip"
    DetectionPaddedRescale = "DetectionPaddedRescale"
    DetectionTargetsFormat = "DetectionTargetsFormat"
    DetectionTargetsFormatTransform = "DetectionTargetsFormatTransform"
    RandomResizedCropAndInterpolation = "RandomResizedCropAndInterpolation"
    RandAugmentTransform = "RandAugmentTransform"
    Lighting = "Lighting"
    RandomErase = "RandomErase"


TRANSFORMS = {
    TransformNames.RandomFlipSeg: RandomFlip,
    TransformNames.ResizeSeg: ResizeSeg,
    TransformNames.RescaleSeg: Rescale,
    TransformNames.RandomRescaleSeg: RandomRescale,
    TransformNames.RandomRotateSeg: RandomRotate,
    TransformNames.CropImageAndMaskSeg: CropImageAndMask,
    TransformNames.RandomGaussianBlurSeg: RandomGaussianBlur,
    TransformNames.PadShortToCropSizeSeg: PadShortToCropSize,
    TransformNames.ColorJitterSeg: ColorJitterSeg,
    TransformNames.DetectionMosaic: DetectionMosaic,
    TransformNames.DetectionRandomAffine: DetectionRandomAffine,
    TransformNames.DetectionMixup: DetectionMixup,
    TransformNames.DetectionHSV: DetectionHSV,
    TransformNames.DetectionHorizontalFlip: DetectionHorizontalFlip,
    TransformNames.DetectionPaddedRescale: DetectionPaddedRescale,
    TransformNames.DetectionTargetsFormat: DetectionTargetsFormat,
    TransformNames.DetectionTargetsFormatTransform: DetectionTargetsFormatTransform,
    TransformNames.RandomResizedCropAndInterpolation: RandomResizedCropAndInterpolation,
    TransformNames.RandAugmentTransform: rand_augment_transform,
    TransformNames.Lighting: Lighting,
    TransformNames.RandomErase: RandomErase
}