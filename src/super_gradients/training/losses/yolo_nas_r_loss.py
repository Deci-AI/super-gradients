import dataclasses
import math
from typing import Tuple, List, Optional

import torch
from super_gradients.common.environment.ddp_utils import get_world_size, is_distributed
from super_gradients.common.registry.registry import register_loss
from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import undo_flat_collate_tensors_with_batch_index
from torch import nn, Tensor

from .ppyolo_loss import gather_topk_anchors, compute_max_iou_anchor
from ..models.detection_models.yolo_nas_r.yolo_nas_r_ndfl_heads import YoloNASRLogits


def check_points_inside_rboxes(points: Tensor, rboxes: Tensor) -> Tensor:
    """

    :param points: Tensor (float) of shape[L, 2], "xy" format, L: num_anchors
    :param rboxes: Tensor (float) of shape[B, n, 5], CXCYWHR

    :return is_in_bboxes: Tensor (float) of shape[B, n, L], value=1. means selected
    """
    points = points[None, None, :, :]  # [1, 1, L, 2]
    x, y = points[..., 0], points[..., 1]  # [1, 1, L], [1, 1, L]

    cx, cy, w, h = rboxes[..., 0, None], rboxes[..., 1, None], rboxes[..., 2, None], rboxes[..., 3, None]
    smallest_radius = torch.minimum(w, h) / 2
    smallest_radius_sqr = smallest_radius**2

    distance_sqr = (x - cx).pow(2) + (y - cy).pow(2)  # [B, n, L]
    # check whether distance between points and center of bboxes is less than mean radius of the rotated boxes
    is_in_bboxes: Tensor = distance_sqr <= smallest_radius_sqr  # [B, 1, n, L]
    return is_in_bboxes.type_as(rboxes)


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with cxcywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[..., 2:4].pow(2) / 12, boxes[..., 4:]), dim=-1)
    a, b, c = gbbs[..., 0], gbbs[..., 1], gbbs[..., 2]
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def pairwise_cxcywhr_iou(obb1, obb2, CIoU=False, eps=1e-7):
    obb1 = obb1[..., :, None, :]
    obb2 = obb2[..., None, :, :]
    return cxcywhr_iou(obb1, obb2, CIoU=CIoU, eps=eps)


def cxcywhr_iou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (..., N, 5) representing ground truth boxes, with cxcywhr format.
        obb2 (torch.Tensor): A tensor of shape (..., M, 5) representing predicted boxes, with cxcywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (..., N, M) representing obb similarities.
    """
    x1, y1 = obb1[..., 0], obb1[..., 1]
    x2, y2 = obb2[..., 0], obb2[..., 1]
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd

    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2], obb1[..., 3]
        w2, h2 = obb2[..., 2], obb2[..., 3]

        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU

    return iou


@dataclasses.dataclass
class YoloNASRAssignmentResult:
    """
    This dataclass stores result of assignment of predicted boxes to ground truth boxes for YoloNASPose model.
    It produced by YoloNASPoseTaskAlignedAssigner and is used by YoloNASPoseLoss to compute the loss.

    For all fields, first dimension is batch dimension, second dimension is number of anchors.

    :param assigned_labels: Tensor of shape (B, L) - Assigned gt labels for each anchor location
    :param assigned_rboxes: Tensor of shape (B, L, 5) - Assigned groundtruth boxes in CXCYWHR format for each anchor location
    :param assigned_scores: Tensor of shape (B, L, C) - Assigned scores for each anchor location
    :param assigned_gt_index: Tensor of shape (B, L) - Index of assigned groundtruth box for each anchor location
    :param assigned_crowd: Tensor of shape (B, L) - Whether the assigned groundtruth box is crowd
    """

    assigned_labels: Tensor
    assigned_rboxes: Tensor
    assigned_scores: Tensor
    assigned_gt_index: Tensor
    assigned_crowd: Tensor


class YoloNASRAssigner(nn.Module):
    """
    Task-aligned assigner repurposed from YoloNAS for OBB OD task
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta=6.0, eps=1e-9):
        """

        :param topk:                 Maximum number of anchors that is selected for each gt box
        :param alpha:                Power factor for class probabilities of predicted boxes (Used compute alignment metric)
        :param beta:                 Power factor for IoU score of predicted boxes (Used compute alignment metric)
        :param eps:                  Small constant for numerical stability
        """
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_rboxes: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_rboxes: Tensor,
        gt_crowd: Tensor,
        pad_gt_mask: Optional[Tensor],
        bg_index: int,
    ) -> YoloNASRAssignmentResult:
        """
        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.

        :param pred_scores: Tensor (float32): predicted class probability, shape(B, L, C)
        :param pred_rboxes: Tensor (float32): predicted rotated boxes, shape(B, L, 5) in cxcywhr format
        :param anchor_points: Tensor (float32): pre-defined anchors, shape(L, 2), xy format
               Must be multiplied by stride before passing to this function
        :param gt_labels: Tensor (int64|int32): Label of gt_bboxes, shape(B, n, 1)
        :param gt_rboxes: Tensor (float32): Ground truth bboxes, shape(B, n, 5) in cxcywhr format
        :param gt_crowd: Tensor (int): Whether the gt is crowd, shape(B, n, 1)
        :param pad_gt_mask: Tensor (float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        :param bg_index: int (background index)
        :return:
            - assigned_labels, Tensor of shape (B, L)
            - assigned_bboxes, Tensor of shape (B, L, 4)
            - assigned_scores, Tensor of shape (B, L, C)
        """
        assert pred_scores.ndim == pred_rboxes.ndim
        assert gt_labels.ndim == gt_rboxes.ndim and gt_rboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_rboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.long, device=gt_labels.device)
            assigned_rboxes = torch.zeros([batch_size, num_anchors, 5], device=gt_labels.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=gt_labels.device)
            assigned_gt_index = torch.zeros([batch_size, num_anchors], dtype=torch.long, device=gt_labels.device)
            assigned_crowd = torch.zeros([batch_size, num_anchors], dtype=torch.bool, device=gt_labels.device)

            return YoloNASRAssignmentResult(
                assigned_labels=assigned_labels,
                assigned_rboxes=assigned_rboxes,
                assigned_scores=assigned_scores,
                assigned_gt_index=assigned_gt_index,
                assigned_crowd=assigned_crowd,
            )

        # compute iou between gt and pred bbox, [B, n, L]

        ious = pairwise_cxcywhr_iou(gt_rboxes, pred_rboxes)
        if ious.size(1) != num_max_boxes or ious.size(2) != num_anchors:
            raise ValueError("The shape of ious is not correct.")

        # if gt_labels.min().item() < 0 or gt_labels.max().item() >= num_classes:
        #     raise ValueError(f"The value of gt_labels is not correct. Found values outside of [0, num_classes): {torch.unique(gt_labels)}")

        # gather pred bboxes class score
        pred_scores = torch.permute(pred_scores, [0, 2, 1])  # [B, Anchors, C] -> [B, C, Anchors]
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device).unsqueeze(-1)
        gt_labels_ind = torch.stack([batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)], dim=-1)

        bbox_cls_scores = pred_scores[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]

        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # check the positive sample's center in gt, [B, n, L]
        # is_in_gts = check_points_inside_rboxes(anchor_points, gt_rboxes) do not check
        is_in_gts = torch.ones(alignment_metrics)
        # select top-k alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts
        if pad_gt_mask is not None:
            mask_positive *= pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(gt_labels.flatten(), index=assigned_gt_index.flatten(), dim=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))

        assigned_rboxes = gt_rboxes.reshape([-1, 5])[assigned_gt_index.flatten(), :]
        assigned_rboxes = assigned_rboxes.reshape([batch_size, num_anchors, 5])

        assigned_scores = torch.nn.functional.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(assigned_scores, index=torch.tensor(ind, device=assigned_scores.device, dtype=torch.long), dim=-1)
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = alignment_metrics / (max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        # respect crowd
        assigned_crowd = torch.gather(gt_crowd.flatten(), index=assigned_gt_index.flatten(), dim=0)
        assigned_crowd = assigned_crowd.reshape([batch_size, num_anchors])
        assigned_scores = assigned_scores * assigned_crowd.eq(0).unsqueeze(-1)

        return YoloNASRAssignmentResult(
            assigned_labels=assigned_labels,
            assigned_rboxes=assigned_rboxes,
            assigned_scores=assigned_scores,
            assigned_gt_index=assigned_gt_index,
            assigned_crowd=assigned_crowd,
        )


@register_loss()
class YoloNASRLoss(nn.Module):
    """
    Loss for training YoloNAS-R model
    """

    def __init__(
        self,
        classification_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        dfl_loss_weight: float = 0.1,
        size_loss_weight: float = 1.0,
        centers_loss_weight: float = 1.0,
        bbox_assigner_topk: int = 13,
        bbox_assigned_alpha: float = 1.0,
        bbox_assigned_beta: float = 6.0,
        average_losses_in_ddp: bool = False,
        use_varifocal_loss: bool = True,
    ):
        """
        :param classification_loss_weight: Classification loss weight
        :param iou_loss_weight:            IoU loss weight
        :param dfl_loss_weight:            DFL loss weight
        :param average_losses_in_ddp:      Whether to average losses in DDP mode. In theory, enabling this option
                                           should have the positive impact on model accuracy since it would smooth out
                                           influence of batches with small number of objects.
                                           However, it needs to be proven empirically.
        """
        super().__init__()
        self.use_varifocal_loss = use_varifocal_loss
        self.classification_loss_weight = classification_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.size_loss_weight = size_loss_weight
        self.centers_loss_weight = centers_loss_weight

        self.assigner = YoloNASRAssigner(
            topk=bbox_assigner_topk,
            alpha=bbox_assigned_alpha,
            beta=bbox_assigned_beta,
        )
        self.average_losses_in_ddp = average_losses_in_ddp

    def forward(
        self,
        outputs: YoloNASRLogits,
        targets: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        :param outputs: Tuple of pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor
        :param targets: A tuple of (boxes, labels, crowd) tensors where
                        - boxes: [N, 6] (batch_index, cx, cy, w, h, r)
                        - labels: [N, 2] (batch_index, class_index)
                        - crowd: [N, 2] (batch_index, is_crowd)
        :return:        Tuple of two tensors where first element is main loss for backward and
                        second element is stacked tensor of all individual losses
        """
        batch_size = outputs.score_logits.size(0)
        num_classes = outputs.score_logits.size(2)
        rboxes_list, labels_list, iscrowd_list = self._get_targets_for_sequential_assigner(targets, batch_size=batch_size)

        cls_loss_sum = 0
        iou_loss_sum = 0
        dfl_loss_sum = 0
        centers_l1_loss_sum = 0
        sizes_l1_loss_sum = 0
        assigned_scores_sum_total = 0
        decoded_predictions = outputs.as_decoded()

        for i in range(batch_size):
            with torch.no_grad():
                assign_result = self.assigner(
                    pred_scores=decoded_predictions.scores[i].unsqueeze(0),
                    pred_rboxes=decoded_predictions.boxes_cxcywhr[i].unsqueeze(0),
                    anchor_points=outputs.anchor_points * outputs.strides,
                    gt_labels=labels_list[i].unsqueeze(0),
                    gt_rboxes=rboxes_list[i].unsqueeze(0),
                    gt_crowd=iscrowd_list[i].unsqueeze(0),
                    pad_gt_mask=None,
                    bg_index=num_classes,
                )

            with torch.cuda.amp.autocast(False):
                if self.use_varifocal_loss:
                    one_hot_label = torch.nn.functional.one_hot(assign_result.assigned_labels, num_classes + 1)[..., :-1]
                    cls_loss = self._varifocal_loss(outputs.score_logits[i : i + 1].float(), assign_result.assigned_scores.float(), one_hot_label)
                else:
                    alpha_l = -1
                    cls_loss = self._focal_loss(outputs.score_logits[i : i + 1], assign_result.assigned_scores, alpha_l)

                if not torch.isfinite(cls_loss).all():
                    raise ValueError(
                        "Classification loss is not finite\n"
                        f"score logits is finite: {torch.isfinite(outputs.score_logits).all()}\n"
                        f"labels: {labels_list[i]}\n"
                        f"rboxes: {rboxes_list[i]}\n"
                        f"{outputs.score_logits[i]}\n"
                    )

                loss_iou, loss_dfl, loss_l1_centers, loss_l1_size = self._rbox_loss_v2(
                    pred_dist=outputs.size_dist[i : i + 1],
                    pred_bboxes=decoded_predictions.boxes_cxcywhr[i : i + 1],
                    pred_offsets=outputs.offsets[i : i + 1],
                    anchor_points=outputs.anchor_points,
                    assign_result=assign_result,
                    strides=outputs.strides,
                    reg_max=outputs.reg_max,
                    bg_class_index=num_classes,
                )

                if not torch.isfinite(loss_iou).all():
                    raise ValueError("IoU loss is not finite")
                if not torch.isfinite(loss_dfl).all():
                    raise ValueError("DFL loss is not finite")
                if not torch.isfinite(loss_l1_centers).all():
                    raise ValueError("Centers L1 loss is not finite")
                if not torch.isfinite(loss_l1_size).all():
                    raise ValueError("Sizes L1 loss is not finite")

                cls_loss_sum += cls_loss
                iou_loss_sum += loss_iou
                dfl_loss_sum += loss_dfl
                centers_l1_loss_sum += loss_l1_centers
                sizes_l1_loss_sum += loss_l1_size
                assigned_scores_sum_total += assign_result.assigned_scores.sum()

        if self.average_losses_in_ddp and is_distributed():
            torch.distributed.all_reduce(cls_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(iou_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(dfl_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(centers_l1_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sizes_l1_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(assigned_scores_sum_total, op=torch.distributed.ReduceOp.SUM)
            assigned_scores_sum_total /= get_world_size()

        assigned_scores_sum_total = torch.clip(assigned_scores_sum_total, min=1.0)

        loss_cls = cls_loss_sum * self.classification_loss_weight / assigned_scores_sum_total
        loss_iou = iou_loss_sum * self.iou_loss_weight / assigned_scores_sum_total
        loss_dfl = dfl_loss_sum * self.dfl_loss_weight / assigned_scores_sum_total
        loss_l1_centers = centers_l1_loss_sum * self.centers_loss_weight / assigned_scores_sum_total
        loss_l1_sizes = sizes_l1_loss_sum * self.size_loss_weight / assigned_scores_sum_total

        loss = loss_cls + loss_iou + loss_dfl + loss_l1_centers + loss_l1_sizes
        log_losses = torch.stack([loss_cls.detach(), loss_iou.detach(), loss_dfl.detach(), loss_l1_centers.detach(), loss_l1_sizes.detach(), loss.detach()])

        return loss, log_losses

    @property
    def component_names(self):
        return ["loss_cls", "loss_iou", "loss_dfl", "loss_l1_centers", "loss_l1_sizes", "loss"]

    @torch.no_grad()
    def _get_targets_for_sequential_assigner(self, targets: Tuple[Tensor, Tensor, Tensor], batch_size: int) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Unpack input targets into list of targets for each sample in batch
        :param targets: (N, 6)
        :return: Tuple of two lists. Each list has [batch_size] elements
                 - List of tensors holding class indexes for each target in image
                 - List of tensors holding bbox coordinates (XYXY) for each target in image
        """
        target_bboxes, target_labels, target_crowd = targets

        rboxes_cxcywhr = undo_flat_collate_tensors_with_batch_index(target_bboxes, batch_size)
        labels = undo_flat_collate_tensors_with_batch_index(target_labels, batch_size)
        is_crowd = undo_flat_collate_tensors_with_batch_index(target_crowd, batch_size)

        return rboxes_cxcywhr, labels, is_crowd

    def _df_loss(self, pred_dist: Tensor, target_dist: Tensor) -> Tensor:
        target_left = target_dist.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target_dist
        weight_right = 1 - weight_left

        # [B,L,C] -> [B,C,L] to make compatible with torch.nn.functional.cross_entropy
        # which expects channel dim to be at index 1
        pred_dist = torch.moveaxis(pred_dist, -1, 1)

        loss_left = torch.nn.functional.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
        loss_right = torch.nn.functional.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        return (loss_left + loss_right).mean(dim=-1, keepdim=True)

    def _rbox_loss(
        self, pred_dist, pred_bboxes, pred_offsets, strides, anchor_points, assign_result: YoloNASRAssignmentResult, reg_max: int, bg_class_index: int
    ):
        # select positive samples mask that are not crowd and not background
        # loss ALWAYS respect the crowd targets by excluding them from contributing to the loss
        # if you want to train WITH crowd targets, mark them as non-crowd on dataset level
        # if you want to train WITH crowd targets, mark them as non-crowd on dataset level
        mask_positive = (assign_result.assigned_labels != bg_class_index) * assign_result.assigned_crowd.eq(0)
        num_pos = mask_positive.sum()

        # pos/neg loss
        if num_pos > 0:
            rbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 5])
            size_mask = mask_positive.unsqueeze(-1).tile([1, 1, 2])

            pred_bboxes_pos = torch.masked_select(pred_bboxes, rbox_mask).reshape([-1, 5])
            assigned_bboxes_pos = torch.masked_select(assign_result.assigned_rboxes, rbox_mask).reshape([-1, 5])

            bbox_weight = torch.masked_select(assign_result.assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            iou = cxcywhr_iou(pred_bboxes_pos, assigned_bboxes_pos, CIoU=False)
            loss_iou = 1 - iou
            loss_iou = (loss_iou * bbox_weight.squeeze(-1)).sum()

            dist_mask = mask_positive.unsqueeze(-1).tile([1, 1, (reg_max + 1) * 2])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 2, reg_max + 1])

            assigned_wh_dfl_targets = self._rbox2distance(assign_result.assigned_rboxes, strides, reg_max)
            assigned_wh_dfl_targets_pos = torch.masked_select(assigned_wh_dfl_targets, size_mask).reshape([-1, 2])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_wh_dfl_targets_pos)
            loss_dfl = (loss_dfl * bbox_weight).sum()

            assigned_wh_pos = assigned_bboxes_pos[..., 2:4]
            pred_wh_pos = pred_bboxes_pos[..., 2:4]
            loss_l1_size = torch.nn.functional.smooth_l1_loss(pred_wh_pos, assigned_wh_pos, reduction="none")
            loss_l1_size = (loss_l1_size.mean(dim=-1, keepdim=True) * bbox_weight).sum()

            assigned_cxcy_pos = assigned_bboxes_pos[..., 0:2]
            pred_centers_pos = pred_bboxes_pos[..., 0:2]
            loss_l1_centers = torch.nn.functional.smooth_l1_loss(pred_centers_pos, assigned_cxcy_pos, reduction="none")
            loss_l1_centers = (loss_l1_centers.mean(dim=-1, keepdim=True) * bbox_weight).sum()
        else:
            loss_iou = torch.zeros([], device=pred_bboxes.device)
            loss_dfl = torch.zeros([], device=pred_bboxes.device)
            loss_l1_centers = torch.zeros([], device=pred_bboxes.device)
            loss_l1_size = torch.zeros([], device=pred_bboxes.device)

        return loss_iou, loss_dfl, loss_l1_centers, loss_l1_size

    def _rbox_loss_v2(
        self, pred_dist, pred_bboxes, pred_offsets, strides, anchor_points, assign_result: YoloNASRAssignmentResult, reg_max: int, bg_class_index: int
    ):
        # select positive samples mask that are not crowd and not background
        # loss ALWAYS respect the crowd targets by excluding them from contributing to the loss
        # if you want to train WITH crowd targets, mark them as non-crowd on dataset level
        # if you want to train WITH crowd targets, mark them as non-crowd on dataset level
        mask_positive = (assign_result.assigned_labels != bg_class_index) * assign_result.assigned_crowd.eq(0)  # [B, L]
        bbox_weight = assign_result.assigned_scores.sum(-1) * mask_positive  # [B, L]
        bs = bbox_weight.size(0)
        # IOU
        iou = cxcywhr_iou(pred_bboxes, assign_result.assigned_rboxes, CIoU=False)
        loss_iou = 1 - iou
        loss_iou = (loss_iou * bbox_weight).sum(dtype=torch.float32)

        # DFL
        assigned_wh_dfl_targets = self._rbox2distance(assign_result.assigned_rboxes, strides, reg_max)
        pred_dist = pred_dist.reshape([bs, -1, 2, reg_max + 1])
        loss_dfl = self._df_loss(pred_dist, assigned_wh_dfl_targets)
        loss_dfl = (loss_dfl.squeeze(-1) * bbox_weight).sum(dtype=torch.float32)

        # L1 Size
        loss_l1_size = torch.nn.functional.smooth_l1_loss(pred_bboxes[..., 2:4], assign_result.assigned_rboxes[..., 2:4], reduction="none")
        loss_l1_size = (loss_l1_size.mean(dim=-1, keepdim=False) * bbox_weight).sum(dtype=torch.float32)

        # L1 Centers
        loss_l1_centers = torch.nn.functional.smooth_l1_loss(pred_bboxes[..., 0:2], assign_result.assigned_rboxes[..., 0:2], reduction="none")
        loss_l1_centers = (loss_l1_centers.mean(dim=-1, keepdim=False) * bbox_weight).sum(dtype=torch.float32)

        return loss_iou, loss_dfl, loss_l1_centers, loss_l1_size

    def _rbox2distance(self, rboxes, stride, reg_max: int):
        wh = rboxes[..., 2:4] / stride
        return wh.clip(0, reg_max - 0.01)

    @staticmethod
    def _varifocal_loss(pred_logits: Tensor, gt_score: Tensor, label: Tensor, alpha=0.75, gamma=2.0) -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = weight * torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, gt_score, reduction="none")
        return loss.sum(dtype=torch.float32)

    @staticmethod
    def _focal_loss(pred_logits: Tensor, label: Tensor, alpha=0.25, gamma=2.0, reduction="sum") -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = torch.abs(pred_score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = weight * torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, label, reduction="none")

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Unsupported reduction type {reduction}")
        return loss
