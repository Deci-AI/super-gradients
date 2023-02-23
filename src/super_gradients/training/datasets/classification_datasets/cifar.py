from typing import Optional, Callable, Union

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.decorators.factory_decorator import resolve_param


@register_dataset(Datasets.CIFAR_10)
class Cifar10(CIFAR10):
    """
    CIFAR10 Dataset

    :param root:                    Path for the data to be extracted
    :param train:                   Bool to load training (True) or validation (False) part of the dataset
    :param transforms:              List of transforms to apply sequentially on sample. Wrapped internally with torchvision.Compose
    :param target_transform:        Transform to apply to target output
    :param download:                Download (True) the dataset from source
    """

    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        root: str,
        train: bool = True,
        transforms: Union[list, dict] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        # TO KEEP BACKWARD COMPATABILITY, WILL BE REMOVED IN THE FUTURE ONCE WE ALLIGN TORCHVISION/NATIVE TRANSFORMS
        # TREATMENT IN FACTORIES (I.E STATING COMPOSE IN CONFIGS)
        if isinstance(transforms, list):
            transforms = Compose(transforms)

        super(Cifar10, self).__init__(
            root=root,
            train=train,
            transform=transforms,
            target_transform=target_transform,
            download=download,
        )


@register_dataset(Datasets.CIFAR_100)
class Cifar100(CIFAR100):
    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        root: str,
        train: bool = True,
        transforms: Union[list, dict] = None,
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
        # TO KEEP BACKWARD COMPATABILITY, WILL BE REMOVED IN THE FUTURE ONCE WE ALLIGN TORCHVISION/NATIVE TRANSFORMS
        # TREATMENT IN FACTORIES (I.E STATING COMPOSE IN CONFIGS)
        if isinstance(transforms, list):
            transforms = Compose(transforms)

        super(Cifar100, self).__init__(
            root=root,
            train=train,
            transform=transforms,
            target_transform=target_transform,
            download=download,
        )
