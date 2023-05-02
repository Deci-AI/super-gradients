from typing import Tuple

from torch import Tensor


class RescoringPoseEstimationDecodeCallback:
    """
    A special adapter callback to be used with PoseEstimationMetrics to use the outputs from rescoring model inside metric class.
    """

    def __init__(self, apply_sigmoid: bool):
        """

        :param apply_sigmoid: If True, apply the sigmoid activation on heatmap. This is needed when heatmap is not
                              bound to [0..1] range and trained with logits (E.g focal loss)
        """
        super().__init__()
        self.apply_sigmoid = apply_sigmoid

    def __call__(self, predictions: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """ """
        poses, scores = predictions
        if self.apply_sigmoid:
            scores = scores.sigmoid()
        return poses, scores.squeeze(-1)  # Pose Estimation Callback expects that scores don't have the dummy dimension
