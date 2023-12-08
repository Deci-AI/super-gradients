from typing import Union, Dict

import torchvision.datasets as torch_datasets
from torchvision.transforms import Compose

from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.object_names import Datasets, Processings
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.module_interfaces import HasPreprocessingParams
from super_gradients.training.datasets.classification_datasets.torchvision_utils import get_torchvision_transforms_equivalent_processing


@register_dataset(Datasets.IMAGENET_DATASET)
class ImageNetDataset(torch_datasets.ImageFolder, HasPreprocessingParams):
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

    @resolve_param("transforms", factory=TransformsFactory())
    def __init__(self, root: str, transforms: Union[list, dict] = [], *args, **kwargs):
        # TO KEEP BACKWARD COMPATABILITY, WILL BE REMOVED IN THE FUTURE ONCE WE ALLIGN TORCHVISION/NATIVE TRANSFORMS
        # TREATMENT IN FACTORIES (I.E STATING COMPOSE IN CONFIGS)
        if isinstance(transforms, list):
            transforms = Compose(transforms)
        super(ImageNetDataset, self).__init__(root, transform=transforms, *args, **kwargs)

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
