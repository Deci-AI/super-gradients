from typing import List, Tuple, Optional

import numpy as np
import torch

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample


@register_transform(Transforms.KeypointsImageToTensor)
class KeypointsImageToTensor:
    """
    Convert image from numpy array to tensor and permute axes to [C,H,W].

    This transform works only for old-style transform API and will raise an exception when used in strongly-typed
    data samples transform API.
    """

    def __init__(self):
        self.additional_samples_count = 0

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert image from numpy array to tensor and permute axes to [C,H,W].
        """
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        return image, mask, joints, areas, bboxes

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        raise RuntimeError(
            f"{self.__class__} does not have apply_to_sample method because manual channel permutation HWC->CHW"
            f"is not needed for new data samples API. This is currently performed inside collate_fn."
        )

    def get_equivalent_preprocessing(self) -> List:
        return [
            {Processings.ImagePermute: {"permutation": (2, 0, 1)}},
        ]

    def __repr__(self):
        return self.__class__.__name__ + "()"
