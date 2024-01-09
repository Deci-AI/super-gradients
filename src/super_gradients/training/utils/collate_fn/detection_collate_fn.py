from typing import Tuple, List, Union

import numpy as np
import torch

from super_gradients.common.registry import register_collate_function
from super_gradients.common.exceptions.dataset_exceptions import DatasetItemsException


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
        :return: one tensor of targets of all images of shape [N, 6], where N is the total number of targets in a batch
                 and the 1st column is batch item index
        """
        labels_batch = [torch.tensor(labels) for labels in labels_batch]
        labels_batch_indexed = []
        for i, labels in enumerate(labels_batch):
            batch_column = labels.new_ones((labels.shape[0], 1)) * i
            labels = torch.cat((batch_column, labels), dim=-1)
            labels_batch_indexed.append(labels)
        return torch.cat(labels_batch_indexed, 0)
