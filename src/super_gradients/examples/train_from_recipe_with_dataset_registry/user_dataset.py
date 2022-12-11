from typing import Optional, Callable

from torchvision.datasets import MNIST
from torchvision.transforms import Compose

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.registry.registry import register_dataset


@register_dataset("MnistDataset")
class MnistDataset(MNIST):
    """
    MNIST Dataset

    :param root:                    Path for the data to be extracted
    :param train:                   Bool to load training (True) or validation (False) part of the dataset
    :param transforms:              List of transforms to apply sequentially on sample. Wrapped internally with torchvision.Compose
    :param target_transform:        Transform to apply to target output
    :param download:                Download (True) the dataset from source
    """

    @resolve_param(
        "transforms", ListFactory(TransformsFactory())
    )  # RESOLVE TRANFORMS ARG- NOW WE CAN USE SG TRANFORMS, IN A SIMPLE MANNER THROUGH SPECIFYING THEM IN THE RECIPE YAML FILE (SEE DATASET_PARAMS INSIDE IT)
    def __init__(
        self,
        root: str,
        train: bool = True,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=Compose(transforms),
            target_transform=target_transform,
            download=download,
        )
