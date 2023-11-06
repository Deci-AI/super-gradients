from typing import Optional, Callable, Union, Dict

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose

from super_gradients.common.object_names import Datasets, Processings
from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.module_interfaces import HasPreprocessingParams
from super_gradients.training.datasets.classification_datasets.torchvision_utils import get_torchvision_transforms_equivalent_processing


@register_dataset(Datasets.CIFAR_10)
class Cifar10(CIFAR10, HasPreprocessingParams):
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


@register_dataset(Datasets.CIFAR_100)
class Cifar100(CIFAR100, HasPreprocessingParams):
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
