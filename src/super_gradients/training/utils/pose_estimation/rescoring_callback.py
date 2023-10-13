from typing import Tuple, List

from torch import Tensor

from super_gradients.module_interfaces import PoseEstimationPredictions


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

    def __call__(self, predictions: Tuple[Tensor, Tensor]) -> List[PoseEstimationPredictions]:
        """ """
        poses, scores = predictions
        if self.apply_sigmoid:
            scores = scores.sigmoid()

        return [PoseEstimationPredictions(poses=poses.cpu().numpy(), scores=scores.squeeze(1).cpu().numpy(), bboxes_xyxy=None)]
