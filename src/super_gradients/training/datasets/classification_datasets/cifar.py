from typing import Optional, Callable

from torchvision.transforms import Compose

from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.decorators.factory_decorator import resolve_param
from torchvision.datasets import CIFAR10, CIFAR100


class Cifar10(CIFAR10):
    """
    CIFAR10 Dataset

    :param root:                    Path for the data to be extracted
    :param train:                   Bool to load training (True) or validation (False) part of the dataset
    :param transforms:              List of transforms to apply sequentially on sample. Wrapped internally with torchvision.Compose
    :param target_transform:        Transform to apply to target output
    :param download:                Download (True) the dataset from source
    """
    @resolve_param("transforms", ListFactory(TransformsFactory()))
    def __init__(
        self,
        root: str,
        train: bool = True,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(Cifar10, self).__init__(
            root=root,
            train=train,
            transform=Compose(transforms),
            target_transform=target_transform,
            download=download,
        )


class Cifar100(CIFAR100):
    @resolve_param("transforms", ListFactory(TransformsFactory()))
    def __init__(
        self,
        root: str,
        train: bool = True,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        CIFAR100 Dataset

        :param root:                    Path for the data to be extracted
        :param train:                   Bool to load training (True) or validation (False) part of the dataset
        :param transforms:              List of transforms to apply sequentially on sample. Wrapped internally with torchvision.Compose
        :param target_transform:        Transform to apply to target output
        :param download:                Download (True) the dataset from source
        """
        super(Cifar100, self).__init__(
            root=root,
            train=train,
            transform=Compose(transforms),
            target_transform=target_transform,
            download=download,
        )
