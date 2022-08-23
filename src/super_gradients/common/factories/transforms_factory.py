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
import os

import numpy as np

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset, BatchSampler, DataLoader
import torchvision.transforms as transforms


from super_gradients.common import DatasetDataInterface
from super_gradients.common.environment import AWS_ENV_NAME
from super_gradients.common.abstractions.abstract_logger import get_logger

from super_gradients.training import utils as core_utils
from super_gradients.training.utils.distributed_training_utils import get_local_rank, wait_for_the_master

from super_gradients.training.utils import get_param

from super_gradients.training.datasets import datasets_utils, DataAugmentation
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.datasets.data_augmentation import Lighting, RandomErase
from super_gradients.training.datasets.mixup import CollateMixup
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset, PascalVOCDetectionDataset

from super_gradients.training.datasets.samplers.infinite_sampler import InfiniteSampler
from super_gradients.training.datasets.segmentation_datasets import PascalVOC2012SegmentationDataSet, \
    PascalAUG2012SegmentationDataSet, CoCoSegmentationDataSet
from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CityscapesDataset
from super_gradients.training.datasets.segmentation_datasets.supervisely_persons_segmentation import \
    SuperviselyPersonsDataset

from super_gradients.training.datasets.samplers.repeated_augmentation_sampler import RepeatAugSampler
from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation, worker_init_reset_seed

from super_gradients.training.exceptions.dataset_exceptions import IllegalDatasetParameterException

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
            'RandomHorizontalFlip': transforms.RandomHorizontalFlip,
            'color_augmentation': datasets_utils.get_color_augmentation,
            'ToTensor': transforms.ToTensor,
            'Lighting': Lighting,
            'Normalize': transforms.Normalize,
            'Resize': transforms.Resize,
            'CenterCrop': transforms.CenterCrop,

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
