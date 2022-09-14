from super_gradients.training.datasets.data_augmentation import Lighting, RandomErase
from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation, rand_augment_transform
from super_gradients.training.transforms.transforms import SegRandomFlip, SegRescale, SegRandomRescale, SegRandomRotate, \
    SegCropImageAndMask, SegRandomGaussianBlur, SegPadShortToCropSize, SegResize, SegColorJitter, DetectionMosaic, DetectionRandomAffine, \
    DetectionMixup, DetectionHSV, \
    DetectionHorizontalFlip, DetectionTargetsFormat, DetectionPaddedRescale, \
    DetectionTargetsFormatTransform
from torchvision.transforms import Compose, ToTensor, PILToTensor, ConvertImageDtype, ToPILImage, Normalize, Resize, CenterCrop, Pad, Lambda, RandomApply,\
    RandomChoice, RandomOrder, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, FiveCrop, TenCrop, LinearTransformation, ColorJitter,\
    RandomRotation, RandomAffine, Grayscale, RandomGrayscale, RandomPerspective, RandomErasing, GaussianBlur, InterpolationMode, RandomInvert, RandomPosterize,\
    RandomSolarize, RandomAdjustSharpness, RandomAutocontrast, RandomEqualize


class TransformNames:
    """Static class holding all the supported transform names"""""
    # From SG
    SegRandomFlip = "SegRandomFlip"
    SegResize = "SegResize"
    SegRescale = "SegRescale"
    SegRandomRescale = "SegRandomRescale"
    SegRandomRotate = "SegRandomRotate"
    SegCropImageAndMask = "SegCropImageAndMask"
    SegRandomGaussianBlur = "SegRandomGaussianBlur"
    SegPadShortToCropSize = "SegPadShortToCropSize"
    SegColorJitter = "SegColorJitter"
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

    # From torch
    Compose = "Compose"
    ToTensor = "ToTensor"
    PILToTensor = "PILToTensor"
    ConvertImageDtype = "ConvertImageDtype"
    ToPILImage = "ToPILImage"
    Normalize = "Normalize"
    Resize = "Resize"
    CenterCrop = "CenterCrop"
    Pad = "Pad"
    Lambda = "Lambda"
    RandomApply = "RandomApply"
    RandomChoice = "RandomChoice"
    RandomOrder = "RandomOrder"
    RandomCrop = "RandomCrop"
    RandomHorizontalFlip = "RandomHorizontalFlip"
    RandomVerticalFlip = "RandomVerticalFlip"
    RandomResizedCrop = "RandomResizedCrop"
    FiveCrop = "FiveCrop"
    TenCrop = "TenCrop"
    LinearTransformation = "LinearTransformation"
    ColorJitter = "ColorJitter"
    RandomRotation = "RandomRotation"
    RandomAffine = "RandomAffine"
    Grayscale = "Grayscale"
    RandomGrayscale = "RandomGrayscale"
    RandomPerspective = "RandomPerspective"
    RandomErasing = "RandomErasing"
    GaussianBlur = "GaussianBlur"
    InterpolationMode = "InterpolationMode"
    RandomInvert = "RandomInvert"
    RandomPosterize = "RandomPosterize"
    RandomSolarize = "RandomSolarize"
    RandomAdjustSharpness = "RandomAdjustSharpness"
    RandomAutocontrast = "RandomAutocontrast"
    RandomEqualize = "RandomEqualize"


TRANSFORMS = {
    TransformNames.SegRandomFlip: SegRandomFlip,
    TransformNames.SegResize: SegResize,
    TransformNames.SegRescale: SegRescale,
    TransformNames.SegRandomRescale: SegRandomRescale,
    TransformNames.SegRandomRotate: SegRandomRotate,
    TransformNames.SegCropImageAndMask: SegCropImageAndMask,
    TransformNames.SegRandomGaussianBlur: SegRandomGaussianBlur,
    TransformNames.SegPadShortToCropSize: SegPadShortToCropSize,
    TransformNames.SegColorJitter: SegColorJitter,
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
    TransformNames.RandomErase: RandomErase,

    # From torch
    TransformNames.Compose: Compose,
    TransformNames.ToTensor: ToTensor,
    TransformNames.PILToTensor: PILToTensor,
    TransformNames.ConvertImageDtype: ConvertImageDtype,
    TransformNames.ToPILImage: ToPILImage,
    TransformNames.Normalize: Normalize,
    TransformNames.Resize: Resize,
    TransformNames.CenterCrop: CenterCrop,
    TransformNames.Pad: Pad,
    TransformNames.Lambda: Lambda,
    TransformNames.RandomApply: RandomApply,
    TransformNames.RandomChoice: RandomChoice,
    TransformNames.RandomOrder: RandomOrder,
    TransformNames.RandomCrop: RandomCrop,
    TransformNames.RandomHorizontalFlip: RandomHorizontalFlip,
    TransformNames.RandomVerticalFlip: RandomVerticalFlip,
    TransformNames.RandomResizedCrop: RandomResizedCrop,
    TransformNames.FiveCrop: FiveCrop,
    TransformNames.TenCrop: TenCrop,
    TransformNames.LinearTransformation: LinearTransformation,
    TransformNames.ColorJitter: ColorJitter,
    TransformNames.RandomRotation: RandomRotation,
    TransformNames.RandomAffine: RandomAffine,
    TransformNames.Grayscale: Grayscale,
    TransformNames.RandomGrayscale: RandomGrayscale,
    TransformNames.RandomPerspective: RandomPerspective,
    TransformNames.RandomErasing: RandomErasing,
    TransformNames.GaussianBlur: GaussianBlur,
    TransformNames.InterpolationMode: InterpolationMode,
    TransformNames.RandomInvert: RandomInvert,
    TransformNames.RandomPosterize: RandomPosterize,
    TransformNames.RandomSolarize: RandomSolarize,
    TransformNames.RandomAdjustSharpness: RandomAdjustSharpness,
    TransformNames.RandomAutocontrast: RandomAutocontrast,
    TransformNames.RandomEqualize: RandomEqualize,
}