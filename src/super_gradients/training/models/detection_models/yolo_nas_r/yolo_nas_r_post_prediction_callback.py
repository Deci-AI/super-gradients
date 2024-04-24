from typing import List

import torch
from torch import Tensor

from super_gradients.module_interfaces.obb_predictions import OBBPredictions, AbstractOBBPostPredictionCallback
from super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_ndfl_heads import YoloNASRLogits


def rboxes_nms(rboxes_cxcywhr: Tensor, scores: Tensor, iou_threshold: float):
    """
    Perform NMS on rotated boxes.
    :param rboxes_cxcywhr: [N,5] Rotated boxes in CXCYWHR format
    :param scores: [N] Confidence scores
    :param iou_threshold: IOU threshold for NMS
    :return: Indices of boxes to keep
    """
    from super_gradients.training.losses.yolo_nas_r_loss import cxcywhr_iou

    idxs = torch.argsort(scores)
    pick = []
    device = rboxes_cxcywhr.device

    # keep looping while some indexes still remain in the indexes
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compute the ratio of overlap
        iou = cxcywhr_iou(rboxes_cxcywhr[i : i + 1], rboxes_cxcywhr[idxs[:last]])

        overlap_with_high_iou_mask = torch.flatten(torch.nonzero(iou > iou_threshold, as_tuple=False))

        indexes_to_delete = torch.cat((torch.tensor([last], device=device, dtype=int), overlap_with_high_iou_mask))
        idxs = torch.index_select(idxs, 0, torch.tensor([j for j in range(len(idxs)) if j not in indexes_to_delete], dtype=int, device=device))

    # return the indicies of the picked bounding boxes that were picked
    return torch.tensor(pick, dtype=torch.int, device=device)


class YoloNASRPostPredictionCallback(AbstractOBBPostPredictionCallback):
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
            pred_rboxes,
            pred_scores,
        ) in zip(predictions.boxes_cxcywhr, predictions.scores):
            # pred_rboxes [Anchors, 5] in CXCYWHR format
            # pred_scores [Anchors, C] confidence scores [0..1]

            pred_cls_conf, pred_cls_label = torch.max(pred_scores, dim=1)

            conf_mask = pred_cls_conf >= self.score_threshold  # [Anchors]

            pred_rboxes = pred_rboxes[conf_mask].float()
            pred_cls_conf = pred_cls_conf[conf_mask]
            pred_cls_label = pred_cls_label[conf_mask]

            # Filter all predictions by self.nms_top_k
            if pred_rboxes.size(0) > self.pre_nms_max_predictions:
                topk_candidates = torch.topk(pred_cls_conf, k=self.pre_nms_max_predictions, largest=True, sorted=True)
                pred_cls_conf = pred_cls_conf[topk_candidates.indices]
                pred_cls_label = pred_cls_label[topk_candidates.indices]

            # NMS
            idx_to_keep = rboxes_nms(rboxes_cxcywhr=pred_rboxes, scores=pred_cls_conf, iou_threshold=self.nms_iou_threshold)

            pred_rboxes = pred_rboxes[idx_to_keep]  # [Instances,]
            pred_cls_conf = pred_cls_conf[idx_to_keep]  # [Instances,]
            pred_cls_label = pred_cls_label[idx_to_keep]  # [Instances,]

            decoded_predictions.append(
                OBBPredictions(
                    scores=pred_cls_conf[: self.post_nms_max_predictions],
                    labels=pred_cls_label[: self.post_nms_max_predictions],
                    rboxes_cxcywhr=pred_rboxes[: self.post_nms_max_predictions],
                )
            )

        return decoded_predictions
