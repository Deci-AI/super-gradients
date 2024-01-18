import itertools
import random
import typing
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
        self,
        random_resize_sizes: Union[List[int], None] = None,
        random_resize_modes: Union[List[int], None] = None,
        random_aspect_ratio: Union[bool, Tuple[float, float]] = False,
    ):
        """
        :param random_resize_sizes: List of single image size dimensions to use for sampling the output image size.
                                    If None, random resizing will not be applied.
                                    If not None, will randomly sample output shape for entire batch:
                                    [B, C, random.choice(random_resize_sizes), random.choice(random_resize_sizes)]
                                    The values in random_resize_sizes should be compatible with the model.
                                    Example: If the model requires input size to be divisible by 32 then all values in `random_resize_sizes`
                                    should be divisible by 32.

        :param random_resize_modes: List of interpolation modes to randomly resize the image to. If None, will not resize.
                                    Interpolation modes correspond to OpenCV interpolation modes:
                                    0 - INTER_NEAREST
                                    1 - INTER_LINEAR
                                    2 - INTER_CUBIC
                                    3 - INTER_AREA
                                    4 - INTER_LANCZOS4
                                    If None defaults to linear interpolation.

        :param random_aspect_ratio: If True, will randomly choose both width and height from random_resize_sizes.
                                    If False, will randomly choose only value which will be the width and height of the images.
                                    If tuple (min_aspect_ratio, max_aspect_ratio), will guarantee that sampled width and height
                                    satisfy required aspect ratio range.
        """
        super().__init__()
        if random_resize_sizes is not None:
            # All possible combinations
            random_resize_sizes = np.array(list(itertools.product(random_resize_sizes, random_resize_sizes)))  # [N, 2]
            if random_aspect_ratio is False:
                # Leave only square sizes
                random_resize_sizes = random_resize_sizes[random_resize_sizes[:, 0] == random_resize_sizes[:, 1]]
            elif random_aspect_ratio is True:
                # No action needed here
                pass
            elif isinstance(random_aspect_ratio, typing.Iterable):
                min_aspect_ratio, max_aspect_ratio = random_aspect_ratio
                if min_aspect_ratio > max_aspect_ratio:
                    raise ValueError(f"min_aspect_ratio: {min_aspect_ratio} must be smaller than max_aspect_ratio: {max_aspect_ratio}")

                # Leave only size combinations with aspect ratio in the given range
                aspect_ratios = random_resize_sizes[:, 0] / random_resize_sizes[:, 1]
                random_resize_sizes = random_resize_sizes[(aspect_ratios >= min_aspect_ratio) & (aspect_ratios <= max_aspect_ratio)]

                if len(random_resize_sizes) == 0:
                    raise ValueError(
                        f"Given random_aspect_ratio value: {random_aspect_ratio} leaves no valid size combinations. Please adjust random_aspect_ratio range."
                    )
            else:
                raise ValueError(f"Unsupported random_aspect_ratio value: {random_aspect_ratio}")
        self.random_resize_sizes = random_resize_sizes
        self.random_resize_modes = list(random_resize_modes) if random_resize_modes is not None else [1]  # Default to linear interpolation

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

    def random_resize_sample(self, sample, target_size: Tuple[int, int], interpolation: int):
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
