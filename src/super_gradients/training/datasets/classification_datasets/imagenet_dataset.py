from typing import Union

import torchvision.datasets as torch_datasets
from torchvision.transforms import Compose

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory


class ImageNetDataset(torch_datasets.ImageFolder):
    """ImageNetDataset dataset. Download here: https://image-net.org/download.php


    Dataset Structure:
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
    """

    @resolve_param("transforms", factory=TransformsFactory())
    def __init__(self, root: str, transforms: Union[list, dict] = [], *args, **kwargs):
        # TO KEEP BACKWARD COMPATABILITY, WILL BE REMOVED IN THE FUTURE ONCE WE ALLIGN TORCHVISION/NATIVE TRANSFORMS
        # TREATMENT IN FACTORIES (I.E STATING COMPOSE IN CONFIGS)
        if isinstance(transforms, list):
            transforms = Compose(transforms)
        super(ImageNetDataset, self).__init__(root, transform=transforms, *args, **kwargs)
