from typing import Tuple, List, Union

import numpy as np
import torch

from super_gradients.common.registry import register_collate_function
from super_gradients.common.exceptions.dataset_exceptions import DatasetItemsException


@register_collate_function()
class OpticalFlowCollateFN:
    """
    Collate function for optical flow training
    """

    def __init__(self):
        self.expected_item_names = ("images", "targets")

    def __call__(self, data) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        try:
            images_batch, labels_batch = list(zip(*data))
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=data[0], collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_labels(labels_batch)

    @staticmethod
    def _format_images(images_batch: List[Union[torch.Tensor, np.array]]) -> torch.Tensor:
        images_batch = [torch.tensor(img) for img in images_batch]
        images_batch_stack = torch.stack(images_batch, 0)
        return images_batch_stack

    @staticmethod
    def _format_labels(labels_batch: List[Union[torch.Tensor, np.array]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split labels to flow maps and valid tensors
        :param labels_batch: a list of targets per image
        :return: a tuple of two tensors of targets of all images, where one tensor is the flow map of shape [2, H, W]
        and another tensor is the valid map of shape [H, W]
        """
        flow_labels_batch = [torch.tensor(flow) for flow, _ in labels_batch]
        valid_labels_batch = [torch.tensor(valid) for _, valid in labels_batch]
        flow_labels_batch_stack = torch.stack(flow_labels_batch, 0)
        valid_labels_batch_stack = torch.stack(valid_labels_batch, 0)
        return flow_labels_batch_stack, valid_labels_batch_stack
