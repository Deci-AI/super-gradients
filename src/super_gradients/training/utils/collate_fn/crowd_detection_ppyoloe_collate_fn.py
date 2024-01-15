from typing import Union, List, Tuple, Dict

import torch

from super_gradients.common.registry import register_collate_function
from super_gradients.common.exceptions.dataset_exceptions import DatasetItemsException
from super_gradients.training.utils.collate_fn.ppyoloe_collate_fn import PPYoloECollateFN


@register_collate_function()
class CrowdDetectionPPYoloECollateFN(PPYoloECollateFN):
    """
    Collate function for Yolox training with additional_batch_items that includes crowd targets
    """

    def __init__(
        self, random_resize_sizes: Union[List[int], None] = None, random_resize_modes: Union[List[int], None] = None, random_aspect_ratio: bool = False
    ):
        """
        :param random_resize_sizes: List of sizes to randomly resize the image to. If None, will not resize.
        :param random_resize_modes: List of interpolation modes to randomly resize the image to. If None, will not resize.
        :param random_aspect_ratio: If True, will randomly choose both width and height from random_resize_sizes.
                                    If False, will randomly choose only value which will be the width and height of the images.
        """
        super().__init__(random_resize_sizes, random_resize_modes, random_aspect_ratio)
        self.expected_item_names = ("image", "targets", "crowd_targets")

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.random_resize_sizes is not None:
            data = self.random_resize(data)

        try:
            images_batch, labels_batch, crowd_labels_batch = list(zip(*data))
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=data[0], collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_targets(labels_batch), {"crowd_targets": self._format_targets(crowd_labels_batch)}
