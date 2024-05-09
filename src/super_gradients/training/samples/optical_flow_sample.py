import dataclasses

import numpy as np


__all__ = ["OpticalFlowSample"]


@dataclasses.dataclass
class OpticalFlowSample:
    """
    A dataclass representing a single optical flow sample.
    Contains input images and flow map.

    :param images:              np.ndarray, Image of [2, H, W, C] shape.
    :param flow_map:            Depth map of [H, W, 2] shape.
    :param valid:               Valid map of [H, W] shape.
    """

    __slots__ = ["images", "flow_map", "valid"]

    images: np.ndarray
    flow_map: np.ndarray
    valid: np.ndarray

    def __init__(self, images: np.ndarray, flow_map: np.ndarray, valid: np.ndarray = None):
        # small sanity check
        dm_shape = flow_map.shape

        if len(dm_shape) == 4:
            if dm_shape[-1] == 1:
                flow_map = np.squeeze(flow_map, axis=-1)
            else:
                raise RuntimeError(f"Flow map should contain only H and W dimensions for both u and v axises, got {len(dm_shape)} dimensions instead.")

        self.images = images
        self.flow_map = flow_map
        self.valid = valid
