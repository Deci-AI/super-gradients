from torch.utils.data.dataloader import default_collate
import random
from typing import List, Union, Tuple, Optional, Dict, Type, Callable
import cv2
import torch
import numpy as np

from super_gradients.common.registry.registry import register_collate_function


from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter


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


class CollateFN:
    def __init__(
        self,
        collage_fn: Optional[Callable] = None,
        adapter: Optional[BaseDatasetAdapter] = None,
        post_process_fn: Optional[Callable] = None,
    ):
        self._collate_fn = collage_fn or default_collate
        self._adapter = adapter
        self._post_process_fn = post_process_fn

    def __call__(self, data):
        batch_data = self._collate_fn(data)
        if self._adapter is not None:
            batch_data = self._adapter.adapt_batch(batch_data)
        if self._post_process_fn is not None:
            batch_data = self._post_process_fn(batch_data)
        return batch_data

    def set_adapter(self, adapter: BaseDatasetAdapter):
        self._adapter = adapter


@register_collate_function()
class DetectionCollateFN(CollateFN):
    """
    Collate function for Yolox training
    """

    def __init__(self, adapter: Optional[DetectionDatasetAdapter] = None):
        self.expected_item_names = ("image", "targets")
        super().__init__(collage_fn=default_collate, adapter=adapter, post_process_fn=self._format_batch)

    def _format_batch(self, batch_data):
        try:
            images_batch, labels_batch = batch_data
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=batch_data[0], collate_type=type(self), expected_item_names=self.expected_item_names)
        return self._format_images(images_batch), self._format_targets(labels_batch)

    @staticmethod
    def _format_images(images_batch: List[Union[torch.Tensor, np.array]]) -> torch.Tensor:
        images_batch = [torch.tensor(img) for img in images_batch]
        images_batch_stack = torch.stack(images_batch, 0)
        if images_batch_stack.shape[3] == 3:
            images_batch_stack = torch.moveaxis(images_batch_stack, -1, 1).float()
        return images_batch_stack

    @staticmethod
    def _format_targets(labels_batch: List[Union[torch.Tensor, np.array]]) -> torch.Tensor:
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

    def __init__(self, adapter: Optional[DetectionDatasetAdapter] = None):
        self.expected_item_names = ("image", "targets", "crowd_targets")
        super().__init__(adapter=adapter)

    def _format_batch(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        try:
            images_batch, labels_batch, crowd_labels_batch = batch_data
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=batch_data[0], collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_targets(labels_batch), {"crowd_targets": self._format_targets(crowd_labels_batch)}


class PPYoloECollateFN(DetectionCollateFN):
    """
    Collate function for PPYoloE training
    """

    def __init__(
        self,
        random_resize_sizes: Union[List[int], None] = None,
        random_resize_modes: Union[List[int], None] = None,
        adapter: Optional[DetectionDatasetAdapter] = None,
    ):
        """
        :param random_resize_sizes: (rows, cols)
        """
        super().__init__(adapter=adapter)
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
