import copy
from typing import Union, Dict, Tuple, Any

import torchvision.datasets as torch_datasets
from torchvision.transforms import Compose

from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.object_names import Processings
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.module_interfaces import HasPreprocessingParams
from super_gradients.training.datasets.classification_datasets.torchvision_utils import get_torchvision_transforms_equivalent_processing


@register_dataset("ImageNetAEDataset")
class ImageNetAEDataset(torch_datasets.ImageFolder, HasPreprocessingParams):
    """ImageNetDataset dataset.

    To use this Dataset you need to:

    - Download imagenet dataset (https://image-net.org/download.php)
        Imagenet
         ├──train
         │  ├──n02093991
         │  │   ├──n02093991_1001.JPEG
         │  │   ├──n02093991_1004.JPEG
         │  │   └──...
         │  ├──n02093992
         │  └──...
         └──val
            ├──n02093991
            ├──n02093992
            └──...

    - Instantiate the dataset:
        >> train_set = ImageNetDataset(root='.../Imagenet/train', ...)
        >> valid_set = ImageNetDataset(root='.../Imagenet/val', ...)
    """

    @resolve_param("pre_transforms", factory=TransformsFactory())
    @resolve_param("augmentations", factory=TransformsFactory())
    @resolve_param("post_transforms", factory=TransformsFactory())
    def __init__(
        self, root: str, pre_transforms: Union[list, dict] = [], augmentations: Union[list, dict] = [], post_transforms: Union[list, dict] = [], *args, **kwargs
    ):
        self.pre_transforms = copy.deepcopy(pre_transforms)
        self.augmentations = copy.deepcopy(augmentations)
        self.post_transforms = copy.deepcopy(post_transforms)
        transforms = self.pre_transforms + self.augmentations + self.post_transforms

        # TO KEEP BACKWARD COMPATABILITY, WILL BE REMOVED IN THE FUTURE ONCE WE ALLIGN TORCHVISION/NATIVE TRANSFORMS
        # TREATMENT IN FACTORIES (I.E STATING COMPOSE IN CONFIGS)
        if isinstance(self.pre_transforms, list):
            self.pre_transforms = Compose(self.pre_transforms)
        if isinstance(self.post_transforms, list):
            self.post_transforms = Compose(self.post_transforms)
        if isinstance(self.augmentations, list):
            self.augmentations = Compose(self.augmentations)

        if isinstance(transforms, list):
            transforms = Compose(transforms)
        super(ImageNetAEDataset, self).__init__(root, transform=transforms, *args, **kwargs)

    def get_dataset_preprocessing_params(self) -> Dict:
        """
        Get the preprocessing params for the dataset.
        It infers preprocessing params from transforms used in the dataset & class names
        :return: (dict) Preprocessing params
        """

        pipeline = get_torchvision_transforms_equivalent_processing(self.transforms)
        params = dict(
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            class_names=self.classes,
        )
        return params

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        original_sample = copy.deepcopy(sample)

        if self.pre_transforms is not None:
            sample = self.pre_transforms(sample)
            original_sample = self.pre_transforms(original_sample)

        if self.augmentations is not None:
            sample = self.augmentations(sample)

        if self.post_transforms is not None:
            sample = self.post_transforms(sample)
            original_sample = self.post_transforms(original_sample)

        return sample, original_sample
