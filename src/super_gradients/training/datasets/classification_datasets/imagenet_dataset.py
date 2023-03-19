from typing import Union

import torchvision.datasets as torch_datasets
from torchvision.transforms import Compose

from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.object_names import Datasets
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory


@register_dataset(Datasets.IMAGENET_DATASET)
class ImageNetDataset(torch_datasets.ImageFolder):
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
