import dataclasses

import numpy as np


__all__ = ["DepthEstimationSample"]


@dataclasses.dataclass
class DepthEstimationSample:
    """
    A dataclass representing a single depth estimation sample.
    Contains input image and depth map.

    :param image:              np.ndarray, Image of [H, W, (C if colorful)] shape.
    :param depth_map:          Depth map of [H, W] shape.
    """

    __slots__ = ["image", "depth_map"]

    image: np.ndarray
    depth_map: np.ndarray

    def __init__(self, image: np.ndarray, depth_map: np.ndarray):
        # small sanity check
        dm_shape = depth_map.shape

        if len(dm_shape) == 3:
            if dm_shape[-1] == 1:
                depth_map = np.squeeze(depth_map, axis=-1)
            else:
                raise RuntimeError(f"Depth map should contain only H and W dimensions, got {len(dm_shape)} dimensions instead.")

        self.image = image
        self.depth_map = depth_map
