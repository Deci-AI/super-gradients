from typing import List, Tuple

import torch
import torchvision

from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback, xyxy2cxcywh


class PPYoloEPostPredictionCallback(DetectionPostPredictionCallback):
    """Non-Maximum Suppression (NMS) module"""

    def __init__(self, conf: float = 0.001, iou: float = 0.6, classes: List[int] = None, max_predictions: int = 300, with_confidence: bool = True):
        """
        :param conf: confidence threshold
        :param iou: IoU threshold                                       (used in NMS_Type.ITERATIVE)
        :param classes: (optional list) filter by class                 (used in NMS_Type.ITERATIVE)
        :param nms_type: the type of nms to use (iterative or matrix)
        :param max_predictions: maximum number of boxes to output       (used in NMS_Type.MATRIX)

        """
        super(PPYoloEPostPredictionCallback, self).__init__()
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.max_pred = max_predictions
        self.with_confidence = with_confidence

    def forward(self, predictions, device: str, image_shape: Tuple[int, int]):
        """

        :param x: Tuple of (bboxes, scores) of shape [B, Anchors, 4], [B, Anchors, C]
        :param device:
        :return:
        """
        nms_result = []
        for pred_bboxes, pred_scores in zip(*predictions):
            # pred_bboxes [Anchors, C],
            # pred_scores [Anchors, 4]
            pred_cls_conf, pred_cls_label = torch.max(pred_scores, dim=1)
            conf_mask = pred_cls_conf >= self.conf

            pred_cls_conf = pred_cls_conf[conf_mask]
            pred_cls_label = pred_cls_label[conf_mask]
            pred_bboxes = pred_bboxes[conf_mask, :]

            idx_to_keep = torchvision.ops.boxes.batched_nms(pred_bboxes, pred_cls_conf, pred_cls_label, self.iou)

            pred_cls_conf = pred_cls_conf[idx_to_keep].unsqueeze(-1)
            pred_cls_label = pred_cls_label[idx_to_keep].unsqueeze(-1)
            pred_bboxes = xyxy2cxcywh(pred_bboxes[idx_to_keep].clone())
            # TODO: Normalize bboxes wrt image shape

            #  nx6 (x1, y1, x2, y2, confidence, class) where x and y are in range [0,1]
            final_boxes = torch.cat([pred_bboxes, pred_cls_conf, pred_cls_label], dim=1)  # [N,6]

            nms_result.append(final_boxes)

        return self._filter_max_predictions(nms_result)

    def _filter_max_predictions(self, res: List) -> List:
        res[:] = [im[: self.max_pred] if (im is not None and im.shape[0] > self.max_pred) else im for im in res]

        return res
