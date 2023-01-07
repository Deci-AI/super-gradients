from typing import Callable, Dict, Tuple, Any, List, Sequence, Optional

import numpy as np
import torch.utils.data.dataset
import torchvision.transforms as T

from torch.utils.data.dataset import Dataset

from super_gradients.training.datasets.datasets_utils import get_random_items
from super_gradients.training.transforms.transforms import Transform, DictTransform
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.datasets_factory import DatasetsFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory


class CustomDataset(Dataset):
    """Wrap any dataset to support SG transforms."""

    @resolve_param("transforms", factory=TransformsFactory())
    @resolve_param("dataset", factory=DatasetsFactory())
    def __init__(self, dataset, transforms: List[Transform], fields: Sequence[str], input_adapters: Optional[Dict[str, Callable]] = None):
        """
        :param dataset:         Dataset to wrap.
        :param transforms:      Transforms
        :param input_adapters:  Dictionary of adapters, that takes every field from dataset.__getitem__ and converts them to the type expected by the transforms
        :param fields:          Name of the output fields of the dataset.
                                    - The order should match the output of the wrapped dataset
                                    - The names should match the fields used in the transform ("image", "target", "mask", ...)
                                    Example:    IF      field = ("image", "target", "crowd_target")
                                                THEN    dataset.__getitem__ should fit the pattern: "image, target, crowd_target = dataset.__getitem__()"
        """
        self.dataset = dataset
        self.transforms = transforms
        self.input_adapters = input_adapters
        self.fields = fields

    def __getitem__(self, item: Tuple[Any]) -> Tuple[np.ndarray]:
        """Wraps the original dataset and apply SG transform over it."""
        items = self.dataset[item]
        if len(items) != len(self.fields):
            raise RuntimeError(f"The dataset {self.dataset.__name__} is expected to return {len(self.fields)} items, but instead returns {len(items)}")

        sample = self._adapt_inputs(items)
        sample = self._apply_transforms(sample)

        return tuple(sample[field] for field in self.fields)

    def _adapt_inputs(self, items: Tuple[Any]) -> Dict[str, np.ndarray]:
        """Transform the dataset output items into a dictionary of np.ndarray.
        This is required in order to apply SG transforms on the dataset items."""
        sample = _tuple_to_dict(self.fields, items)
        for field in self.fields:
            adapter = self.input_adapters.get(field)
            if adapter:
                sample[field] = adapter(sample[field])
        return sample

    def _apply_transforms(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for transform in self.transforms:
            if hasattr(transform, "additional_samples_count"):
                # Get raw samples, but without any Transform
                additional_items = get_random_items(dataset=self.dataset, count=transform.additional_samples_count)
                additional_samples = [self._adapt_inputs(items) for items in additional_items]
                sample = transform(sample={**sample, "additional_samples": additional_samples})
            else:
                sample = transform(sample=sample)
        return sample

    def __len__(self):
        return len(self.dataset)


class CustomDetectionDataset(CustomDataset):
    """Wrap any detection dataset to support DetectionTransforms."""

    def __init__(
        self,
        dataset,
        transforms: List[Transform],
        image_adapter: Optional[Callable] = None,
        target_adapter: Optional[Callable] = None,
    ):
        """
        :param dataset:         Dataset to wrap. Has to fit the pattern: "image, target = dataset.__getitem__()"
        :param transforms:      Transforms
        :param image_adapter:   Adapter function to apply on dataset.__getitem__()[0] to make it fit the expected type of image in the DetectionTransforms
        :param target_adapter:  Adapter function to apply on dataset.__getitem__()[1] to make it fit the expected type of target in the DetectionTransforms
        """
        super().__init__(
            dataset=dataset,
            transforms=transforms,
            fields=("image", "target"),
            input_adapters={"image": image_adapter, "target": target_adapter},
        )


class CustomSegmentationDataset(CustomDataset):
    """Wrap any detection dataset to support SegmentationTransforms."""

    def __init__(
        self,
        dataset,
        transforms: List[Transform],
        image_adapter: Optional[Callable] = None,
        mask_adapter: Optional[Callable] = None,
    ):
        """
        :param dataset:         Dataset to wrap. Has to fit the pattern: "image, mask = dataset.__getitem__()"
        :param transforms:      Transforms
        :param image_adapter:   Adapter function to apply on dataset.__getitem__()[0] to make it fit the expected type of image in the DetectionTransforms
        :param mask_adapter:    Adapter function to apply on dataset.__getitem__()[1] to make it fit the expected type of mask in the DetectionTransforms
        """
        transforms = transforms + [self._get_post_processing_transform()]
        super().__init__(
            dataset=dataset,
            transforms=transforms,
            fields=("image", "mask"),
            input_adapters={"image": image_adapter, "mask": mask_adapter},
        )

    def _get_post_processing_transform(self) -> Transform:
        """Motivation: All the SegmentationDatasets currently implement some transforms that are applied before the collate function behind the hood.
        This transform mimics this behavior to ensure compatibility with SegmentationDatasets.
        """

        def _post_processing_mask(mask: np.ndarray) -> np.ndarray:
            mask = torch.from_numpy(np.array(mask)).long()
            mask[mask == 255] = 19
            return mask

        post_processing = DictTransform(
            {"image": T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), "mask": _post_processing_mask}
        )
        return post_processing


def _tuple_to_dict(fields: Sequence[str], items: Tuple[Any]) -> Dict[str, Any]:
    """Sequentially map items and field to form a dictionary"""
    return {field: val for field, val in zip(fields, items)}
