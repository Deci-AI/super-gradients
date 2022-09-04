import inspect
from typing import Union, Mapping

from omegaconf import ListConfig
from torchvision import transforms

from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.training.datasets.data_augmentation import Lighting, RandomErase
from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation, rand_augment_transform
from super_gradients.training.transforms.transforms import RandomFlip, Rescale, RandomRescale, RandomRotate, \
    CropImageAndMask, RandomGaussianBlur, PadShortToCropSize, ResizeSeg, ColorJitterSeg, DetectionMosaic, DetectionRandomAffine, \
    DetectionMixup, DetectionHSV, \
    DetectionHorizontalFlip, DetectionTargetsFormat, DetectionPaddedRescale, \
    DetectionTargetsFormatTransform


class TransformsFactory(BaseFactory):

    def __init__(self):
        type_dict = {
            'RandomFlipSeg': RandomFlip,
            'ResizeSeg': ResizeSeg,
            'RescaleSeg': Rescale,
            'RandomRescaleSeg': RandomRescale,
            'RandomRotateSeg': RandomRotate,
            'CropImageAndMaskSeg': CropImageAndMask,
            'RandomGaussianBlurSeg': RandomGaussianBlur,
            'PadShortToCropSizeSeg': PadShortToCropSize,
            'ColorJitterSeg': ColorJitterSeg,
            "DetectionMosaic": DetectionMosaic,
            "DetectionRandomAffine": DetectionRandomAffine,
            "DetectionMixup": DetectionMixup,
            "DetectionHSV": DetectionHSV,
            "DetectionHorizontalFlip": DetectionHorizontalFlip,
            "DetectionPaddedRescale": DetectionPaddedRescale,
            "DetectionTargetsFormat": DetectionTargetsFormat,
            "DetectionTargetsFormatTransform": DetectionTargetsFormatTransform,

            'RandomResizedCropAndInterpolation': RandomResizedCropAndInterpolation,
            'RandAugmentTransform': rand_augment_transform,
            'Lighting': Lighting,
            'RandomErase': RandomErase
        }
        for name, obj in inspect.getmembers(transforms, inspect.isclass):
            if name in type_dict:
                raise RuntimeError(f'key {name} already exists in dictionary')

            type_dict[name] = obj

        super().__init__(type_dict)

    def get(self, conf: Union[str, dict]):

        # SPECIAL HANDLING FOR COMPOSE
        if isinstance(conf, Mapping) and 'Compose' in conf:
            conf['Compose']['transforms'] = ListFactory(TransformsFactory()).get(conf['Compose']['transforms'])
        elif isinstance(conf, (list, ListConfig)):
            conf = ListFactory(TransformsFactory()).get(conf)

        return super().get(conf)
