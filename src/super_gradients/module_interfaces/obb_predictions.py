import abc
import dataclasses
from typing import Any, List
from typing import Union

import numpy as np
from torch import Tensor

__all__ = ["OBBPredictions", "AbstractOBBPostPredictionCallback"]


@dataclasses.dataclass
class OBBPredictions:
    """
    A data class that encapsulates oriented box predictions for a single image.

    :param labels: Array of shape [N] with class indices
    :param scores: Array of shape [N] with corresponding confidence scores.
    :param rboxes_cxcywhr: Array of shape [N, 5] with rotated boxes for each pose in CXCYWHR format.
    """

    scores: Union[Tensor, np.ndarray]
    labels: Union[Tensor, np.ndarray]
    rboxes_cxcywhr: Union[Tensor, np.ndarray]

    def __init__(self, rboxes_cxcywhr, scores, labels):
        if len(rboxes_cxcywhr) != len(scores) or len(rboxes_cxcywhr) != len(labels):
            raise ValueError(f"rboxes_cxcywhr, scores and labels must have the same length. Got: {len(rboxes_cxcywhr)}, {len(scores)}, {len(labels)}")
        if rboxes_cxcywhr.ndim != 2 or rboxes_cxcywhr.shape[1] != 5:
            raise ValueError(f"rboxes_cxcywhr must have shape [N, 5]. Got: {rboxes_cxcywhr.shape}")

        self.scores = scores
        self.labels = labels
        self.rboxes_cxcywhr = rboxes_cxcywhr

    def __len__(self):
        return len(self.scores)


class AbstractOBBPostPredictionCallback(abc.ABC):
    """
    A protocol interface of a post-prediction callback for pose estimation models.
    """

    @abc.abstractmethod
    def __call__(self, predictions: Any) -> List[OBBPredictions]:
        ...
