import dataclasses
from typing import Mapping, Tuple, Union, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from super_gradients.common.registry.registry import register_loss
from super_gradients.common.environment.ddp_utils import get_world_size, is_distributed

from super_gradients.training.utils.bbox_utils import batch_distance2bbox

from .ppyolo_loss import GIoULoss, batch_iou_similarity, check_points_inside_bboxes, gather_topk_anchors, compute_max_iou_anchor

from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import undo_flat_collate_tensors_with_batch_index
from ..models.detection_models.yolo_nas_r.yolo_nas_r_ndfl_heads import YoloNASRLogits


def batch_cxcywhr_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Calculate IOU of rotated boxes in batch using Probabilistic IoU (Prob-IoU) approach.
    TODO: DEBUG ME

    Bboxes are expected to be in [cx cy w h r] format.

    :param box1: box with the shape [N, M1, 5]
    :param box2: box with the shape [N, M2, 5]
    :return iou: iou between box1 and box2 with the shape [N, M1, M2]
    """
    N, M1, _ = box1.shape
    _, M2, _ = box2.shape

    # Unpack boxes into coordinates and angles
    box1_x, box1_y, box1_w, box1_h, box1_r = torch.split(box1, 1, dim=-1)
    box2_x, box2_y, box2_w, box2_h, box2_r = torch.split(box2, 1, dim=-1)

    # Calculate the minimum and maximum corners of the boxes
    box1_min_x, box1_max_x, box1_min_y, box1_max_y = calculate_box_min_max(box1_x, box1_y, box1_w, box1_h, box1_r)
    box2_min_x, box2_max_x, box2_min_y, box2_max_y = calculate_box_min_max(box2_x, box2_y, box2_w, box2_h, box2_r)

    # Calculate intersection areas
    inter_width = torch.clamp(torch.min(box1_max_x, box2_max_x) - torch.max(box1_min_x, box2_min_x), min=0)
    inter_height = torch.clamp(torch.min(box1_max_y, box2_max_y) - torch.max(box1_min_y, box2_min_y), min=0)
    intersection = inter_width * inter_height

    # Calculate union areas
    area1 = box1_w * box1_h
    area2 = box2_w * box2_h
    union = area1 + area2 - intersection

    # Calculate Prob-IoU
    iou = torch.clamp(intersection / (union + eps), min=0.0, max=1.0)

    return iou


def calculate_box_min_max(cx, cy, w, h, r):
    """Calculate the minimum and maximum coordinates of the box corners.
    TODO: DEBUG ME
    """
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    dx = w / 2 * cos_r
    dy = h / 2 * sin_r

    x_corners = torch.stack([cx - dx, cx + dx, cx + dx, cx - dx], dim=-1)
    y_corners = torch.stack([cy - dy, cy - dy, cy + dy, cy + dy], dim=-1)

    # Rotate the corners around the center
    x_corners_rot = cx + (x_corners - cx) * cos_r - (y_corners - cy) * sin_r
    y_corners_rot = cy + (x_corners - cx) * sin_r + (y_corners - cy) * cos_r

    min_x = torch.min(x_corners_rot, dim=-1).values
    max_x = torch.max(x_corners_rot, dim=-1).values
    min_y = torch.min(y_corners_rot, dim=-1).values
    max_y = torch.max(y_corners_rot, dim=-1).values

    return min_x, max_x, min_y, max_y


@dataclasses.dataclass
class YoloNASOBBBoxesAssignmentResult:
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


class YoloNASOBBAssigner(nn.Module):
    """
    Task-aligned assigner repurposed from YoloNAS for OBB OD task
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta=6.0, eps=1e-9):
        """

        :param sigmas:               Sigmas for OKS calculation
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
        gt_bboxes: Tensor,
        gt_poses: Tensor,
        gt_crowd: Tensor,
        pad_gt_mask: Tensor,
        bg_index: int,
    ) -> YoloNASOBBBoxesAssignmentResult:
        """
        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.

        :param pred_scores: Tensor (float32): predicted class probability, shape(B, L, C)
        :param pred_rboxes: Tensor (float32): predicted bounding boxes, shape(B, L, 5)
        :param anchor_points: Tensor (float32): pre-defined anchors, shape(L, 2), xy format
        :param gt_labels: Tensor (int64|int32): Label of gt_bboxes, shape(B, n, 1)
        :param gt_bboxes: Tensor (float32): Ground truth bboxes, shape(B, n, 4)
        :param gt_poses: Tensor (float32): Ground truth poses, shape(B, n, Num Keypoints, 3)
        :param gt_crowd: Tensor (int): Whether the gt is crowd, shape(B, n, 1)
        :param pad_gt_mask: Tensor (float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        :param bg_index: int ( background index
        :param gt_scores: Tensor (one, float32) Score of gt_bboxes, shape(B, n, 1)
        :return:
            - assigned_labels, Tensor of shape (B, L)
            - assigned_bboxes, Tensor of shape (B, L, 4)
            - assigned_scores, Tensor of shape (B, L, C)
        """
        assert pred_scores.ndim == pred_rboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.long, device=gt_labels.device)
            assigned_rboxes = torch.zeros([batch_size, num_anchors, 5], device=gt_labels.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=gt_labels.device)
            assigned_gt_index = torch.zeros([batch_size, num_anchors], dtype=torch.long, device=gt_labels.device)
            assigned_crowd = torch.zeros([batch_size, num_anchors], dtype=torch.bool, device=gt_labels.device)

            return YoloNASOBBBoxesAssignmentResult(
                assigned_labels=assigned_labels,
                assigned_rboxes=assigned_rboxes,
                assigned_scores=assigned_scores,
                assigned_gt_index=assigned_gt_index,
                assigned_crowd=assigned_crowd,
            )

        # compute iou between gt and pred bbox, [B, n, L]
        ious = batch_cxcywhr_iou(gt_bboxes, pred_rboxes)

        # gather pred bboxes class score
        pred_scores = torch.permute(pred_scores, [0, 2, 1])
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device).unsqueeze(-1)
        gt_labels_ind = torch.stack([batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)], dim=-1)

        bbox_cls_scores = pred_scores[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]

        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

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

        assigned_rboxes = gt_bboxes.reshape([-1, 5])[assigned_gt_index.flatten(), :]
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

        return YoloNASOBBBoxesAssignmentResult(
            assigned_labels=assigned_labels,
            assigned_rboxes=assigned_rboxes,
            assigned_scores=assigned_scores,
            assigned_gt_index=assigned_gt_index,
            assigned_crowd=assigned_crowd,
        )


@register_loss()
class YoloNASOBBLoss(nn.Module):
    """
    Loss for training YoloNAS-R model
    """

    def __init__(
        self,
        classification_loss_type: str = "focal",
        regression_iou_loss_type: str = "ciou",
        classification_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        dfl_loss_weight: float = 0.5,
        bbox_assigner_topk: int = 13,
        bbox_assigned_alpha: float = 1.0,
        bbox_assigned_beta: float = 6.0,
        average_losses_in_ddp: bool = False,
    ):
        """
        :param oks_sigmas:                 OKS sigmas for pose estimation. Array of [Num Keypoints].
        :param classification_loss_type:   Classification loss type. One of "focal" or "bce"
        :param regression_iou_loss_type:   Regression IoU loss type. One of "giou" or "ciou"
        :param classification_loss_weight: Classification loss weight
        :param iou_loss_weight:            IoU loss weight
        :param dfl_loss_weight:            DFL loss weight
        :param pose_cls_loss_weight:       Pose classification loss weight
        :param pose_reg_loss_weight:       Pose regression loss weight
        :param average_losses_in_ddp:      Whether to average losses in DDP mode. In theory, enabling this option
                                           should have the positive impact on model accuracy since it would smooth out
                                           influence of batches with small number of objects.
                                           However, it needs to be proven empirically.
        """
        super().__init__()
        self.classification_loss_type = classification_loss_type
        self.classification_loss_weight = classification_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.iou_loss_weight = iou_loss_weight

        self.iou_loss = {"giou": GIoULoss, "ciou": CIoULoss}[regression_iou_loss_type]()
        self.num_classes = 1  # We have only one class in pose estimation task
        self.assigner = YoloNASOBBAssigner(
            topk=bbox_assigner_topk,
            alpha=bbox_assigned_alpha,
            beta=bbox_assigned_beta,
        )
        self.average_losses_in_ddp = average_losses_in_ddp

    @torch.no_grad()
    def _unpack_flat_targets(self, targets: Tuple[Tensor, Tensor, Tensor], batch_size: int) -> Mapping[str, torch.Tensor]:
        """
        Convert targets to PPYoloE-compatible format since it's the easiest (not the cleanest) way to
        have PP Yolo training & metrics computed

        :param targets: Tuple (boxes, joints, crowd)
                        - boxes: [N, 5] (batch_index, x1, y1, x2, y2)
                        - joints: [N, num_joints, 4] (batch_index, x, y, visibility)
                        - crowd: [N, 2] (batch_index, is_crowd)
        :return:        (Dictionary [str,Tensor]) with keys:
                        - gt_class: (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
                        - gt_bbox: (Tensor, float32): Ground truth bboxes, shape(B, n, 4) in XYXY format
                        - pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        """
        target_boxes, target_joints, target_iscrowd = targets

        image_index = target_boxes[:, 0]
        gt_bbox = target_boxes[:, 1:5]

        per_image_class = []
        per_image_bbox = []
        per_image_pad_mask = []
        per_image_targets = undo_flat_collate_tensors_with_batch_index(target_joints, batch_size)
        per_image_crowds = undo_flat_collate_tensors_with_batch_index(target_iscrowd, batch_size)

        max_boxes = 0
        for i in range(batch_size):
            mask = image_index == i

            image_bboxes = gt_bbox[mask, :]
            valid_bboxes = image_bboxes.sum(dim=1, keepdims=True) > 0

            per_image_bbox.append(image_bboxes)
            per_image_pad_mask.append(valid_bboxes)
            # Since for pose estimation we have only one class, we can just fill it with zeros
            per_image_class.append(torch.zeros((len(image_bboxes), 1), dtype=torch.long, device=target_boxes.device))

            max_boxes = max(max_boxes, mask.sum().item())

        for i in range(batch_size):
            elements_to_pad = max_boxes - len(per_image_bbox[i])
            padding_left = 0
            padding_right = 0
            padding_top = 0
            padding_bottom = elements_to_pad
            pad = padding_left, padding_right, padding_top, padding_bottom
            per_image_class[i] = F.pad(per_image_class[i], pad, mode="constant", value=0)
            per_image_bbox[i] = F.pad(per_image_bbox[i], pad, mode="constant", value=0)
            per_image_pad_mask[i] = F.pad(per_image_pad_mask[i], pad, mode="constant", value=0)
            per_image_targets[i] = F.pad(per_image_targets[i], (0, 0) + pad, mode="constant", value=0)
            per_image_crowds[i] = F.pad(per_image_crowds[i], pad, mode="constant", value=0)

        new_targets = {
            "gt_class": torch.stack(per_image_class, dim=0),
            "gt_bbox": torch.stack(per_image_bbox, dim=0),
            "pad_gt_mask": torch.stack(per_image_pad_mask, dim=0),
            "gt_poses": torch.stack(per_image_targets, dim=0),
            "gt_crowd": torch.stack(per_image_crowds, dim=0),
        }
        return new_targets

    def forward(
        self,
        outputs: YoloNASRLogits,
        targets: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        :param outputs: Tuple of pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor
        :param targets: A tuple of (boxes, joints, crowd) tensors where
                        - boxes: [N, 5] (batch_index, x1, y1, x2, y2)
                        - joints: [N, num_joints, 4] (batch_index, x, y, visibility)
                        - crowd: [N, 2] (batch_index, is_crowd)
        :return:        Tuple of two tensors where first element is main loss for backward and
                        second element is stacked tensor of all individual losses
        """

        targets = self._unpack_flat_targets(targets, batch_size=outputs.score_logits.size(0))

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes, reg_max = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = targets["gt_class"]
        gt_bboxes = targets["gt_bbox"]
        gt_poses = targets["gt_poses"]
        gt_crowd = targets["gt_crowd"]
        pad_gt_mask = targets["pad_gt_mask"]

        # label assignment
        assign_result = self.assigner(
            pred_scores=pred_scores.detach().sigmoid(),  # Pred scores are logits on training for numerical stability
            pred_bboxes=pred_bboxes.detach() * stride_tensor,
            anchor_points=anchor_points,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            gt_poses=gt_poses,
            gt_crowd=gt_crowd,
            pad_gt_mask=pad_gt_mask,
            bg_index=self.num_classes,
        )

        assigned_scores = assign_result.assigned_scores

        # cls loss
        if self.classification_loss_type == "focal":
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha=-1)
        elif self.classification_loss_type == "bce":
            loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(pred_scores, assigned_scores, reduction="sum")
        else:
            raise ValueError(f"Unknown classification loss type: {self.classification_loss_type}")

        assigned_scores_sum = assigned_scores.sum()
        if self.average_losses_in_ddp and is_distributed():
            torch.distributed.all_reduce(assigned_scores_sum, op=torch.distributed.ReduceOp.SUM)
            assigned_scores_sum /= get_world_size()
        assigned_scores_sum = torch.clip(assigned_scores_sum, min=1.0)
        loss_cls /= assigned_scores_sum

        loss_iou, loss_dfl, loss_pose_cls, loss_pose_reg = self._bbox_loss(
            pred_distri,
            pred_bboxes,
            pred_pose_coords=pred_pose_coords,
            pred_pose_logits=pred_pose_logits,
            stride_tensor=stride_tensor,
            anchor_points=anchor_points_s,
            assign_result=assign_result,
            assigned_scores_sum=assigned_scores_sum,
            reg_max=reg_max,
        )

        loss_cls = loss_cls * self.classification_loss_weight
        loss_iou = loss_iou * self.iou_loss_weight
        loss_dfl = loss_dfl * self.dfl_loss_weight
        loss_pose_cls = loss_pose_cls * self.pose_cls_loss_weight
        loss_pose_reg = loss_pose_reg * self.pose_reg_loss_weight

        loss = loss_cls + loss_iou + loss_dfl + loss_pose_cls + loss_pose_reg
        log_losses = torch.stack([loss_cls.detach(), loss_iou.detach(), loss_dfl.detach(), loss_pose_cls.detach(), loss_pose_reg.detach(), loss.detach()])

        return loss, log_losses

    @property
    def component_names(self):
        return ["loss_cls", "loss_iou", "loss_dfl", "loss_pose_cls", "loss_pose_reg", "loss"]

    def _df_loss(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left

        # [B,L,C] -> [B,C,L] to make compatible with torch.nn.functional.cross_entropy
        # which expects channel dim to be at index 1
        pred_dist = torch.moveaxis(pred_dist, -1, 1)

        loss_left = torch.nn.functional.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
        loss_right = torch.nn.functional.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        return (loss_left + loss_right).mean(dim=-1, keepdim=True)

    def _xyxy_box_area(self, boxes):
        """
        :param boxes: [..., 4] (x1, y1, x2, y2)
        :return: [...,1]
        """
        area = (boxes[..., 2:4] - boxes[..., 0:2]).prod(dim=-1, keepdim=True)
        return area

    def _bbox_loss(
        self,
        pred_dist,
        pred_bboxes,
        pred_pose_coords,
        pred_pose_logits,
        stride_tensor,
        anchor_points,
        assign_result: YoloNASOBBBoxesAssignmentResult,
        assigned_scores_sum,
        reg_max: int,
    ):
        # select positive samples mask that are not crowd and not background
        # loss ALWAYS respect the crowd targets by excluding them from contributing to the loss
        # if you want to train WITH crowd targets, mark them as non-crowd on dataset level
        # if you want to train WITH crowd targets, mark them as non-crowd on dataset level
        mask_positive = (assign_result.assigned_labels != self.num_classes) * assign_result.assigned_crowd.eq(0)
        num_pos = mask_positive.sum()
        assigned_bboxes_divided_by_stride = assign_result.assigned_rboxes / stride_tensor

        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])

            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(assigned_bboxes_divided_by_stride, bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos_image_coord = torch.masked_select(assign_result.assigned_rboxes, bbox_mask).reshape([-1, 4])

            bbox_weight = torch.masked_select(assign_result.assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_iou = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).tile([1, 1, (reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes_divided_by_stride, reg_max)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum

        else:
            loss_iou = torch.zeros([], device=pred_bboxes.device)
            loss_dfl = torch.zeros([], device=pred_bboxes.device)

        return loss_iou, loss_dfl

    def _bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor) -> Tuple[Tensor, int]:
        """
        Decode predicted bounding boxes using anchor points and predicted distribution
        :param anchor_points: Anchor locations (center for each point) of [B, L, 2] shape
        :param pred_dist:     Predicted offset distributions of [B, L, 4 * (reg_max + 1)] shape
        :return:              Decoded bounding boxes (XYXY format) of [B, L, 4] shape and reg_max
        """
        b, l, *_ = pred_dist.size()
        pred_dist = torch.softmax(pred_dist.reshape([b, l, 4, -1]), dim=-1)

        reg_max = pred_dist.size(-1) - 1
        proj_conv = torch.linspace(0, reg_max, reg_max + 1, device=pred_dist.device).reshape([1, reg_max + 1, 1, 1])

        pred_dist = torch.nn.functional.conv2d(pred_dist.permute(0, 3, 1, 2), proj_conv).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist), reg_max

    def _bbox2distance(self, points, bbox, reg_max):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], dim=-1).clip(0, reg_max - 0.01)

    @staticmethod
    def _focal_loss(pred_logits: Tensor, label: Tensor, alpha=0.25, gamma=2.0, reduction="sum") -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = torch.abs(pred_score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        # This is same, but binary_cross_entropy_with_logits is faster
        # loss = -weight * (label * torch.nn.functional.logsigmoid(pred_logits) + (1 - label) * torch.nn.functional.logsigmoid(-pred_logits))
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
