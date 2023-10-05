import abc
import dataclasses
import numpy as np

from typing import Any, List
from typing import Union, Optional
from torch import Tensor

__all__ = ["PoseEstimationPredictions", "AbstractPoseEstimationPostPredictionCallback"]


@dataclasses.dataclass
class PoseEstimationPredictions:
    """
    A data class that encapsulates pose estimation predictions for a single image.

    :param poses:   Array of shape [N, K, 3] where N is number of poses and K is number of joints.
                    Last dimension is [x, y, score] where score the confidence score for the specific joint
                    with [0..1] range.
    :param scores:  Array of shape [N] with scores for each pose with [0..1] range.
    :param bboxes:  Array of shape [N, 4] with bounding boxes for each pose in format [x1, y1, x2, y2].
                    Can be None if bounding boxes are not available (for instance, DEKR model does not output boxes).
    """

    poses: Union[Tensor, np.ndarray]
    scores: Union[Tensor, np.ndarray]
    bboxes: Optional[Union[Tensor, np.ndarray]]


class AbstractPoseEstimationPostPredictionCallback(abc.ABC):
    """
    A protocol interface of a post-prediction callback for pose estimation models.
    """

    @abc.abstractmethod
    def __call__(self, predictions: Any) -> List[PoseEstimationPredictions]:
        ...
