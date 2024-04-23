import abc
import dataclasses
import numpy as np

from typing import Any, List
from typing import Union, Optional
from torch import Tensor

__all__ = ["OBBPredictions"]


@dataclasses.dataclass
class OBBPredictions:
    """
    A data class that encapsulates pose estimation predictions for a single image.

    :param scores:       Array of shape [N] with scores for each pose with [0..1] range.
    :param rboxes_cxcywhr:  Array of shape [N, 5] with rotated boxes for each pose in CXCYWHR format.
                         Can be None if bounding boxes are not available (for instance, DEKR model does not output boxes).
    """

    scores: Union[Tensor, np.ndarray]
    rboxes_cxcywhr: Optional[Union[Tensor, np.ndarray]]


class AbstractOBBPostPredictionCallback(abc.ABC):
    """
    A protocol interface of a post-prediction callback for pose estimation models.
    """

    @abc.abstractmethod
    def __call__(self, predictions: Any) -> List[OBBPredictions]:
        ...
