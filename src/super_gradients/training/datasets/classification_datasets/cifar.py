from typing import Optional, Callable

from torchvision.transforms import Compose

from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.decorators.factory_decorator import resolve_param
from torchvision.datasets import CIFAR10, CIFAR100


class Cifar10(CIFAR10):
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
        super(Cifar100, self).__init__(
            root=root,
            train=train,
            transform=Compose(transforms),
            target_transform=target_transform,
            download=download,
        )
