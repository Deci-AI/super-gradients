from typing import Union, Mapping

from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.training.transforms.transforms import RandomFlip, Rescale, RandomRescale, RandomRotate, \
    CropImageAndMask, RandomGaussianBlur, PadShortToCropSize, ColorJitterSeg

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

