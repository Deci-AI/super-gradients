from torch.utils.data.dataloader import default_collate
import random
from typing import List, Union, Tuple, Optional, Dict, Type, Callable, Iterable

from abc import ABC

import cv2
import torch
import numpy as np


from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter
from data_gradients.dataset_adapters.segmentation_adapter import SegmentationDatasetAdapter
from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter

from super_gradients.training.utils.detection_utils import xyxy2cxcywh
from super_gradients.common.registry.registry import register_collate_function

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.collate_functions_factory import CollateFunctionsFactory


class DatasetItemsException(Exception):
    def __init__(self, data_sample: Tuple, collate_type: Type, expected_item_names: Tuple):
        """
        :param data_sample: item(s) returned by a dataset
        :param collate_type: type of the collate that caused the exception
        :param expected_item_names: tuple of names of items that are expected by the collate to be returned from the dataset
        """
        collate_type_name = collate_type.__name__
        num_sample_items = len(data_sample) if isinstance(data_sample, tuple) else 1
        error_msg = f"`{collate_type_name}` only supports Datasets that return a tuple {expected_item_names}, but got a tuple of len={num_sample_items}"
        super().__init__(error_msg)


class BaseDatasetAdapterCollateFN(ABC):
    @resolve_param("collate_fn", CollateFunctionsFactory())
    def __init__(self, adapter: BaseDatasetAdapter, collate_fn: Optional[Callable] = None):
        self._adapter = adapter
        self._collate_fn = collate_fn or default_collate
        self._adapt_on_batch = adapter.data_config.is_batch or adapter.data_config.is_batch is None

    def __call__(self, samples: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._adapt_on_batch:
            _samples = []
            for sample in samples:
                images, targets = self._adapter.adapt(sample)  # Will construct batch of 1
                images, targets = images.squeeze(0), targets.squeeze(0)  # Extract the sample
                _samples.append((images, targets))
            samples = _samples

        batch = self._collate_fn(samples)

        if self._adapt_on_batch:
            batch = self._adapter.adapt(batch)

        images, targets = batch  # At this point we know it is (images, targets) because the adapter was used - either on samples or batch
        return images, targets


@register_collate_function()
class DetectionDatasetAdapterCollateFN(BaseDatasetAdapterCollateFN):
    @resolve_param("collate_fn", CollateFunctionsFactory())
    def __init__(self, cache_path: str, collate_fn: Optional[Callable] = None):
        adapter = DetectionDatasetAdapter(cache_path=cache_path, n_classes=80)
        super().__init__(adapter=adapter, collate_fn=collate_fn)

    def __call__(self, samples: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = super().__call__(samples=samples)  # This already returns a batch of (images, targets)
        images = images / 255
        targets = flatten_bbox_batch(targets)  # (BS, P, 5) -> (N, 6)
        targets[:, 2:] = xyxy2cxcywh(targets[:, 2:])
        return images, targets


@register_collate_function()
class SegmentationDatasetAdapterCollateFN(BaseDatasetAdapterCollateFN):
    @resolve_param("collate_fn", CollateFunctionsFactory())
    def __init__(self, cache_path: str, collate_fn: Optional[Callable] = None):
        adapter = SegmentationDatasetAdapter(cache_path=cache_path, n_classes=6)
        super().__init__(adapter=adapter, collate_fn=collate_fn)

    def __call__(self, samples: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = super().__call__(samples=samples)  # This already returns a batch of (images, targets)
        images = images / 255
        targets = targets.argmax(1)
        return images, targets


@register_collate_function()
class ClassificationDatasetAdapterCollateFN(BaseDatasetAdapterCollateFN):
    @resolve_param("collate_fn", CollateFunctionsFactory())
    def __init__(self, cache_path: str, collate_fn: Optional[Callable] = None):
        adapter = ClassificationDatasetAdapter(cache_path=cache_path, n_classes=80)
        super().__init__(adapter=adapter, collate_fn=collate_fn)

    def __call__(self, samples: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = super().__call__(samples=samples)  # This already returns a batch of (images, targets)
        images = images / 255
        return images, targets


def flatten_bbox_batch(bbox_batch: torch.Tensor) -> torch.Tensor:
    """
    Flatten a batched bounding box tensor and prepend the batch ID to each bounding box. Excludes padding boxes.

    :param bbox_batch: Bounding box tensor of shape (BS, PaddingSize, 5).
    :return: Flattened tensor of shape (N, 6), where N <= BS * PaddingSize.
    """

    # Create a tensor of batch IDs
    batch_ids = torch.arange(bbox_batch.size(0), device=bbox_batch.device).unsqueeze(-1)
    batch_ids = batch_ids.repeat(1, bbox_batch.size(1)).reshape(-1, 1)  # Shape: (BS*PaddingSize, 1)

    # Reshape bounding box tensor
    bbox_reshaped = bbox_batch.reshape(-1, 5)  # Shape: (BS*PaddingSize, 5)

    # Concatenate batch IDs and reshaped bounding boxes
    flat_bbox = torch.cat((batch_ids, bbox_reshaped), dim=1)  # Shape: (BS*PaddingSize, 6)

    # Filter out padding boxes (assuming padding boxes have all values zero)
    non_padding_mask = torch.any(flat_bbox[:, 1:] != 0, dim=1)
    flat_bbox = flat_bbox[non_padding_mask]

    return flat_bbox


@register_collate_function()
class DetectionCollateFN:
    """
    Collate function for Yolox training
    """

    def __init__(self):
        self.expected_item_names = ("image", "targets")

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            images_batch, labels_batch = list(zip(*data))
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=data[0], collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_targets(labels_batch)

    def _format_images(self, images_batch: List[Union[torch.Tensor, np.array]]) -> torch.Tensor:
        images_batch = [torch.tensor(img) for img in images_batch]
        images_batch_stack = torch.stack(images_batch, 0)
        if images_batch_stack.shape[3] == 3:
            images_batch_stack = torch.moveaxis(images_batch_stack, -1, 1).float()
        return images_batch_stack

    def _format_targets(self, labels_batch: List[Union[torch.Tensor, np.array]]) -> torch.Tensor:
        """
        Stack a batch id column to targets and concatenate
        :param labels_batch: a list of targets per image (each of arbitrary length)
        :return: one tensor of targets of all imahes of shape [N, 6], where N is the total number of targets in a batch
                 and the 1st column is batch item index
        """
        labels_batch = [torch.tensor(labels) for labels in labels_batch]
        labels_batch_indexed = []
        for i, labels in enumerate(labels_batch):
            batch_column = labels.new_ones((labels.shape[0], 1)) * i
            labels = torch.cat((batch_column, labels), dim=-1)
            labels_batch_indexed.append(labels)
        return torch.cat(labels_batch_indexed, 0)


@register_collate_function()
class CrowdDetectionCollateFN(DetectionCollateFN):
    """
    Collate function for Yolox training with additional_batch_items that includes crowd targets
    """

    def __init__(self):
        self.expected_item_names = ("image", "targets", "crowd_targets")
        super().__init__()

    def _format_batch(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        try:
            images_batch, labels_batch, crowd_labels_batch = batch_data
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=batch_data, collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_targets(labels_batch), {"crowd_targets": self._format_targets(crowd_labels_batch)}


class PPYoloECollateFN(DetectionCollateFN):
    """
    Collate function for PPYoloE training
    """

    def __init__(self, random_resize_sizes: Union[List[int], None] = None, random_resize_modes: Union[List[int], None] = None):
        """
        :param random_resize_sizes: (rows, cols)
        """
        super().__init__()
        self.random_resize_sizes = random_resize_sizes
        self.random_resize_modes = random_resize_modes

    def __repr__(self):
        return f"PPYoloECollateFN(random_resize_sizes={self.random_resize_sizes}, random_resize_modes={self.random_resize_modes})"

    def __str__(self):
        return self.__repr__()

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.random_resize_sizes is not None:
            data = self.random_resize(data)
        return super().__call__(data)

    def random_resize(self, batch):
        target_size = random.choice(self.random_resize_sizes)
        interpolation = random.choice(self.random_resize_modes)
        batch = [self.random_resize_sample(sample, target_size, interpolation) for sample in batch]
        return batch

    def random_resize_sample(self, sample, target_size, interpolation):
        if len(sample) == 2:
            image, targets = sample  # TARGETS ARE IN LABEL_CXCYWH
            with_crowd = False
        elif len(sample) == 3:
            image, targets, crowd_targets = sample
            with_crowd = True
        else:
            raise DatasetItemsException(data_sample=sample, collate_type=type(self), expected_item_names=self.expected_item_names)

        dsize = int(target_size), int(target_size)
        scale_factors = target_size / image.shape[0], target_size / image.shape[1]

        image = cv2.resize(
            image,
            dsize=dsize,
            interpolation=interpolation,
        )

        sy, sx = scale_factors
        targets[:, 1:5] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
        if with_crowd:
            crowd_targets[:, 1:5] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
            return image, targets, crowd_targets

        return image, targets


class CrowdDetectionPPYoloECollateFN(PPYoloECollateFN):
    """
    Collate function for Yolox training with additional_batch_items that includes crowd targets
    """

    def __init__(self, random_resize_sizes: Union[List[int], None] = None, random_resize_modes: Union[List[int], None] = None):
        super().__init__(random_resize_sizes, random_resize_modes)
        self.expected_item_names = ("image", "targets", "crowd_targets")

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        if self.random_resize_sizes is not None:
            data = self.random_resize(data)

        try:
            images_batch, labels_batch, crowd_labels_batch = list(zip(*data))
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=data[0], collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_targets(labels_batch), {"crowd_targets": self._format_targets(crowd_labels_batch)}
