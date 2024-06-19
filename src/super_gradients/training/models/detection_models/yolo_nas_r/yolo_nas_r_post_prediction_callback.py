from typing import List, Union, Tuple

import torch
from super_gradients.module_interfaces.obb_predictions import OBBPredictions, AbstractOBBPostPredictionCallback
from super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_ndfl_heads import YoloNASRLogits
from torch import Tensor


def rboxes_matrix_nms(
    rboxes_cxcywhr: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_threshold: float,
    already_sorted: bool,
    class_agnostic_nms: bool = False,
    kernel: str = "gaussian",
    sigma: float = 3.0,
) -> Tensor:
    """
    Implementation of NMS method for rotated boxes.
    This implementation uses approximate IoU calculation for rotated boxes based on gaussian bbox representation.

    :param rboxes_cxcywhr: Input rotated boxes [..., N, 5] in CXCYWHR format
    :param scores: Confidence scores for each box [..., N]
    :param labels: Labels for each box [..., N]
    :param iou_threshold: IoU threshold for NMS
    :param already_sorted: If True, input boxes are already sorted by confidence
    :param class_agnostic_nms: If True, NMS will be class agnostic
    :param kernel: Kernel function for NMS. Can be "gaussian" or "linear"
    :param sigma: Sigma parameter for gaussian kernel. Larger sigma will make NMS more aggressive
    :return: Indexes of boxes to keep
    """
    from super_gradients.training.losses.yolo_nas_r_loss import pairwise_cxcywhr_iou

    if rboxes_cxcywhr.shape[0] == 0:
        # Return empty index tensor of [...., N] shape
        shape = list(rboxes_cxcywhr.size())
        return torch.tensor([], device=rboxes_cxcywhr.device, dtype=torch.long).view(*shape[:-1])

    if not already_sorted:
        order_by_conf_desc = torch.argsort(scores, dim=-1, descending=True)
        rboxes_cxcywhr = rboxes_cxcywhr[order_by_conf_desc]
        scores = scores[order_by_conf_desc]
        labels = labels[order_by_conf_desc]

    iou = pairwise_cxcywhr_iou(rboxes_cxcywhr, rboxes_cxcywhr)
    iou = torch.triu(iou, diagonal=1)

    if not class_agnostic_nms:
        # Create a labels mask, we want only boxes with the same label to affect each other
        labels_matrix = (labels[..., None] == labels[..., None, :]).float().triu(1)
        iou *= labels_matrix

    ious_cmax = iou.max(-2).values.unsqueeze(-1)

    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (ious_cmax**2))
        decay = (decay_matrix / compensate_matrix).min(dim=-2).values
    else:
        decay = (1 - iou) / (1 - ious_cmax)
        decay = decay.min(dim=-2).values

    keep = scores * decay > iou_threshold

    if not already_sorted:
        return order_by_conf_desc[keep]
    else:
        return keep


def rboxes_nms(rboxes_cxcywhr: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Implementation of NMS method for rotated boxes.
    This implementation uses approximate IoU calculation for rotated boxes based on gaussian bbox representation.

    :param rboxes_cxcywhr: Input rotated boxes in CXCYWHR format
    :param scores: Confidence scores for each box
    :param iou_threshold: IoU threshold for NMS
    :return: Indexes of boxes to keep
    """
    from super_gradients.training.losses.yolo_nas_r_loss import pairwise_cxcywhr_iou

    order_by_conf_desc = torch.argsort(scores, descending=True)
    rboxes_cxcywhr = rboxes_cxcywhr[order_by_conf_desc]
    device = rboxes_cxcywhr.device
    keep = torch.ones(len(rboxes_cxcywhr), dtype=torch.bool, device=device)
    iou = pairwise_cxcywhr_iou(rboxes_cxcywhr, rboxes_cxcywhr)
    iou = torch.triu(iou, diagonal=1)

    # Compute mask of boxes with overlas greater than threshold
    iou_gt_mask: Tensor = iou > iou_threshold

    for i in range(len(rboxes_cxcywhr)):
        mask = keep & iou_gt_mask[i]
        keep[mask] = False

    return order_by_conf_desc[keep]


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
        output_device="cpu",
        class_agnostic_nms: bool = False,
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
        self.class_agnostic_nms = class_agnostic_nms
        self.pre_nms_max_predictions = pre_nms_max_predictions
        self.post_nms_max_predictions = post_nms_max_predictions
        self.output_device = output_device

    @torch.no_grad()
    def __call__(self, outputs: Union[Tuple[Tensor, Tensor], YoloNASRLogits]) -> List[OBBPredictions]:
        """
        Take YoloNASPose's predictions and decode them into usable pose predictions.

        :param outputs: Output of the model's forward() method
        :return:        List of decoded predictions for each image in the batch.
        """
        if isinstance(outputs, YoloNASRLogits):
            predictions = outputs.as_decoded()
            boxes = predictions.boxes_cxcywhr
            scores = predictions.scores
        else:
            boxes, scores = outputs

        decoded_predictions: List[OBBPredictions] = []
        for (
            pred_rboxes,
            pred_scores,
        ) in zip(boxes, scores):
            # pred_rboxes [Anchors, 5] in CXCYWHR format
            # pred_scores [Anchors, C] confidence scores [0..1]

            pred_cls_conf, pred_cls_label = torch.max(pred_scores, dim=1)

            conf_mask = pred_cls_conf >= self.score_threshold  # [Anchors]

            pred_rboxes = pred_rboxes[conf_mask].float()
            pred_cls_conf = pred_cls_conf[conf_mask].float()
            pred_cls_label = pred_cls_label[conf_mask].float()

            # Filter all predictions by self.nms_top_k
            if pred_rboxes.size(0) > self.pre_nms_max_predictions:
                topk_candidates = torch.topk(pred_cls_conf, k=self.pre_nms_max_predictions, largest=True, sorted=True)
                pred_rboxes = pred_rboxes[topk_candidates.indices]
                pred_cls_conf = pred_cls_conf[topk_candidates.indices]
                pred_cls_label = pred_cls_label[topk_candidates.indices]

            # NMS
            idx_to_keep = rboxes_matrix_nms(
                rboxes_cxcywhr=pred_rboxes,
                scores=pred_cls_conf,
                labels=pred_cls_label,
                iou_threshold=self.nms_iou_threshold,
                already_sorted=False,
                class_agnostic_nms=self.class_agnostic_nms,
            )

            pred_rboxes = pred_rboxes[idx_to_keep]  # [Instances,5]
            pred_cls_conf = pred_cls_conf[idx_to_keep]  # [Instances,]
            pred_cls_label = pred_cls_label[idx_to_keep]  # [Instances,]

            if self.output_device is not None:
                pred_rboxes = pred_rboxes.to(self.output_device)
                pred_cls_conf = pred_cls_conf.to(self.output_device)
                pred_cls_label = pred_cls_label.to(self.output_device)

            p = OBBPredictions(
                scores=pred_cls_conf[: self.post_nms_max_predictions],
                labels=pred_cls_label[: self.post_nms_max_predictions],
                rboxes_cxcywhr=pred_rboxes[: self.post_nms_max_predictions],
            )
            decoded_predictions.append(p)

        return decoded_predictions
