from typing import Union, Mapping

from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.training.transforms.transforms import RandomFlip, Rescale, RandomRescale, RandomRotate, \
    CropImageAndMask, RandomGaussianBlur, PadShortToCropSize, ColorJitterSeg, DetectionMosaic, DetectionRandomAffine, \
    DetectionMixup, DetectionHSV, \
    DetectionHorizontalFlip, DetectionTargetsFormat, DetectionPaddedRescale, \
    DetectionTargetsFormatTransform

from torchvision import transforms
import inspect



from typing import Union, Mapping

from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.training.transforms.transforms import RandomFlip, Rescale, RandomRescale, RandomRotate, \
    CropImageAndMask, RandomGaussianBlur, PadShortToCropSize, ColorJitterSeg

from super_gradients.training.datasets import datasets_utils
from super_gradients.training.datasets.data_augmentation import Lighting, RandomErase

from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation


from torchvision import transforms
import inspect






class TransformsFactory(BaseFactory):

    def __init__(self):
        type_dict = {
            'RandomFlipSeg': RandomFlip,
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
            # 'RandomHorizontalFlip': transforms.RandomHorizontalFlip,
            'color_augmentation': datasets_utils.get_color_augmentation,
            # 'ToTensor': transforms.ToTensor,
            'Lighting': Lighting,
<<<<<<< HEAD
            'Normalize': transforms.Normalize,
            'Resize': transforms.Resize,
            'CenterCrop': transforms.CenterCrop,

=======
            'RandomErase': RandomErase
            # 'Normalize': transforms.Normalize,
            # 'Resize': transforms.Resize,
            # 'CenterCrop': transforms.CenterCrop,
>>>>>>> 55e97b4f (base working)
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
        return super().get(conf)
