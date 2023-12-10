import random
from typing import Union, List, Tuple

import cv2
import numpy as np
import torch

from super_gradients.common.registry import register_collate_function
from super_gradients.common.exceptions.dataset_exceptions import DatasetItemsException
from super_gradients.training.utils.collate_fn.detection_collate_fn import DetectionCollateFN


@register_collate_function()
class PPYoloECollateFN(DetectionCollateFN):
    """
    Collate function for PPYoloE training
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
        super().__init__()
        self.random_resize_sizes = random_resize_sizes
        self.random_resize_modes = random_resize_modes
        self.random_aspect_ratio = random_aspect_ratio

    def __repr__(self):
        return f"PPYoloECollateFN(random_resize_sizes={self.random_resize_sizes}, random_resize_modes={self.random_resize_modes})"

    def __str__(self):
        return self.__repr__()

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.random_resize_sizes is not None:
            data = self.random_resize(data)
        return super().__call__(data)

    def random_resize(self, batch):
        if self.random_aspect_ratio:
            target_size = random.choices(self.random_resize_sizes, k=2)
        else:
            target_size = random.choice(self.random_resize_sizes)
            target_size = target_size, target_size

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

        target_width, target_height = target_size
        dsize = int(target_width), int(target_height)
        scale_factors = target_height / image.shape[0], target_width / image.shape[1]

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
