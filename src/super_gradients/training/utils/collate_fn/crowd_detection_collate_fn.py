from typing import Tuple, Dict

import torch

from super_gradients.common.registry import register_collate_function
from super_gradients.common.exceptions.dataset_exceptions import DatasetItemsException
from super_gradients.training.utils.collate_fn.detection_collate_fn import DetectionCollateFN


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
