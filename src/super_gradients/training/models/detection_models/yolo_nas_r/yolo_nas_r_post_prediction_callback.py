from typing import List

import torch
from torch import Tensor

from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback
from super_gradients.module_interfaces.obb_predictions import OBBPredictions
from super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_ndfl_heads import YoloNASRLogits


def rboxes_nms(rboxes_cxcywhr: Tensor, scores: Tensor, iou_threshold: float):
    """
    Perform NMS on rotated boxes.
    :param rboxes_cxcywhr: [N,5] Rotated boxes in CXCYWHR format
    :param scores: [N] Confidence scores
    :param iou_threshold: IOU threshold for NMS
    :return: Indices of boxes to keep
    """
    raise NotImplementedError("Implement this function")


class YoloNASRPostPredictionCallback(AbstractPoseEstimationPostPredictionCallback):
    """
    A post-prediction callback for YoloNASPose model.
    Performs confidence thresholding, Top-K and NMS steps.
    """

    def __init__(
        self,
        score_threshold: float,
        nms_iou_threshold: float,
        pre_nms_max_predictions: int,
        post_nms_max_predictions: int,
    ):
        """
        :param score_threshold: Detection confidence threshold
        :param nms_iou_threshold:         IoU threshold for NMS step.
        :param pre_nms_max_predictions:   Number of predictions participating in NMS step
        :param post_nms_max_predictions:  Maximum number of boxes to return after NMS step
        """
        if post_nms_max_predictions > pre_nms_max_predictions:
            raise ValueError("post_nms_max_predictions must be less than pre_nms_max_predictions")

        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.pre_nms_max_predictions = pre_nms_max_predictions
        self.post_nms_max_predictions = post_nms_max_predictions

    @torch.no_grad()
    def __call__(self, outputs: YoloNASRLogits) -> List[OBBPredictions]:
        """
        Take YoloNASPose's predictions and decode them into usable pose predictions.

        :param outputs: Output of the model's forward() method
        :return:        List of decoded predictions for each image in the batch.
        """
        # First is model predictions, second element of tuple is logits for loss computation
        predictions = outputs.as_decoded()

        decoded_predictions: List[OBBPredictions] = []
        for (
            pred_rboxes_cxcywhr,
            pred_bboxes_conf,
        ) in zip(predictions.boxes_cxcywhr, predictions.scores):
            # pred_bboxes [Anchors, 5] in CXCYWHR format
            # pred_scores [Anchors, 1] confidence scores [0..1]

            pred_bboxes_conf = pred_bboxes_conf.squeeze(-1)  # [Anchors]
            conf_mask = pred_bboxes_conf >= self.score_threshold  # [Anchors]

            pred_bboxes_conf = pred_bboxes_conf[conf_mask].float()
            pred_rboxes_cxcywhr = pred_rboxes_cxcywhr[conf_mask].float()

            # Filter all predictions by self.nms_top_k
            if pred_bboxes_conf.size(0) > self.pre_nms_max_predictions:
                topk_candidates = torch.topk(pred_bboxes_conf, k=self.pre_nms_max_predictions, largest=True, sorted=True)
                pred_bboxes_conf = pred_bboxes_conf[topk_candidates.indices]
                pred_rboxes_cxcywhr = pred_rboxes_cxcywhr[topk_candidates.indices]

            # NMS
            idx_to_keep = rboxes_nms(rboxes_cxcywhr=pred_rboxes_cxcywhr, scores=pred_bboxes_conf, iou_threshold=self.nms_iou_threshold)

            final_rboxes = pred_rboxes_cxcywhr[idx_to_keep]  # [Instances,]
            final_scores = pred_bboxes_conf[idx_to_keep]  # [Instances,]

            decoded_predictions.append(
                OBBPredictions(
                    scores=final_scores[: self.post_nms_max_predictions],
                    rboxes_cxcywhr=final_rboxes[: self.post_nms_max_predictions],
                )
            )

        return decoded_predictions
