from typing import List

import torch
import torchvision

from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback


class PPYoloEPostPredictionCallback(DetectionPostPredictionCallback):
    """Non-Maximum Suppression (NMS) module"""

    def __init__(self, score_threshold: float, nms_threshold: float, nms_top_k: int, max_predictions: int, multi_label_per_box: bool = True):
        """
        :param score_threshold: Predictions confidence threshold. Predictions with score lower than score_threshold will not participate in Top-K & NMS
        :param iou: IoU threshold for NMS step.
        :param nms_top_k: Number of predictions participating in NMS step
        :param max_predictions: maximum number of boxes to return after NMS step

        """
        super(PPYoloEPostPredictionCallback, self).__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.max_predictions = max_predictions
        self.multi_label_per_box = multi_label_per_box

    def forward(self, outputs, device: str):
        """

        :param x: Tuple of (bboxes, scores) of shape [B, Anchors, 4], [B, Anchors, C]
        :param device:
        :return:
        """
        nms_result = []
        # First is model predictions, second element of tuple is logits for loss computation
        predictions = outputs[0]

        for pred_bboxes, pred_scores in zip(*predictions):
            # pred_bboxes [Anchors, 4],
            # pred_scores [Anchors, C]

            # Filter all predictions by self.score_threshold
            if self.multi_label_per_box:
                i, j = (pred_scores > self.score_threshold).nonzero(as_tuple=False).T
                pred_bboxes = pred_bboxes[i]
                pred_cls_conf = pred_scores[i, j]
                pred_cls_label = j[:]

            else:
                pred_cls_conf, pred_cls_label = torch.max(pred_scores, dim=1)
                conf_mask = pred_cls_conf >= self.score_threshold

                pred_cls_conf = pred_cls_conf[conf_mask]
                pred_cls_label = pred_cls_label[conf_mask]
                pred_bboxes = pred_bboxes[conf_mask, :]

            # Filter all predictions by self.nms_top_k
            if pred_cls_conf.size(0) > self.nms_top_k:
                topk_candidates = torch.topk(pred_cls_conf, k=self.nms_top_k, largest=True)
                pred_cls_conf = pred_cls_conf[topk_candidates.indices]
                pred_cls_label = pred_cls_label[topk_candidates.indices]
                pred_bboxes = pred_bboxes[topk_candidates.indices, :]

            # NMS
            idx_to_keep = torchvision.ops.boxes.batched_nms(boxes=pred_bboxes, scores=pred_cls_conf, idxs=pred_cls_label, iou_threshold=self.nms_threshold)

            pred_cls_conf = pred_cls_conf[idx_to_keep].unsqueeze(-1)
            pred_cls_label = pred_cls_label[idx_to_keep].unsqueeze(-1)
            pred_bboxes = pred_bboxes[idx_to_keep, :]

            #  nx6 (x1, y1, x2, y2, confidence, class) in pixel units
            final_boxes = torch.cat([pred_bboxes, pred_cls_conf, pred_cls_label], dim=1)  # [N,6]

            nms_result.append(final_boxes)

        return self._filter_max_predictions(nms_result)

    def _filter_max_predictions(self, res: List) -> List:
        res[:] = [im[: self.max_predictions] if (im is not None and im.shape[0] > self.max_predictions) else im for im in res]

        return res
