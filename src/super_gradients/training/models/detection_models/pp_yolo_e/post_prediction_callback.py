from typing import List

from super_gradients.training.models.detection_models.pp_yolo_e.nms import MultiClassNMS
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback


class PPYoloEPostPredictionCallback(DetectionPostPredictionCallback):
    """Non-Maximum Suppression (NMS) module"""

    def __init__(self, conf: float = 0.001, iou: float = 0.6, classes: List[int] = None, max_predictions: int = 300, with_confidence: bool = True):
        """
        :param conf: confidence threshold
        :param iou: IoU threshold                                       (used in NMS_Type.ITERATIVE)
        :param classes: (optional list) filter by class                 (used in NMS_Type.ITERATIVE)
        :param nms_type: the type of nms to use (iterative or matrix)
        :param max_predictions: maximum number of boxes to output       (used in NMS_Type.MATRIX)
        :param with_confidence: in NMS, whether to multiply objectness  (used in NMS_Type.ITERATIVE)
                                score with class score
        """
        super(PPYoloEPostPredictionCallback, self).__init__()
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.nms = MultiClassNMS()
        self.max_pred = max_predictions
        self.with_confidence = with_confidence

    def forward(self, x, device: str = None):
        """

        :param x: Tuple of (bboxes, scores) of shape [B, Anchors, 4], [B, Anchors, C]
        :param device:
        :return:
        """
        pred_bboxes, pred_scores = x  # [B, C, Anchors], [B, Anchors, 4]
        nms_result = self.nms(x)
        return self._filter_max_predictions(nms_result)

    def _filter_max_predictions(self, res: List) -> List:
        res[:] = [im[: self.max_pred] if (im is not None and im.shape[0] > self.max_pred) else im for im in res]

        return res
