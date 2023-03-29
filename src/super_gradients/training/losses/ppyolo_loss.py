from typing import Mapping, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import super_gradients
from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.training.datasets.data_formats.bbox_formats.cxcywh import cxcywh_to_xyxy
from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from super_gradients.training.utils.distributed_training_utils import (
    get_world_size,
)


def batch_iou_similarity(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-9) -> float:
    """Calculate iou of box1 and box2 in batch. Bboxes are expected to be in x1y1x2y2 format.

    :param box1: box with the shape [N, M1, 4]
    :param box2: box with the shape [N, M2, 4]
    :return iou: iou between box1 and box2 with the shape [N, M1, M2]

    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def iou_similarity(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Calculate iou of box1 and box2. Bboxes are expected to be in x1y1x2y2 format.

    :param box1: box with the shape [M1, 4]
    :param box2: box with the shape [M2, 4]

    :return iou: iou between box1 and box2 with the shape [M1, M2]
    """
    box1 = box1.unsqueeze(1)  # [M1, 4] -> [M1, 1, 4]
    box2 = box2.unsqueeze(0)  # [M2, 4] -> [1, M2, 4]
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_overlaps(bboxes1: torch.Tensor, bboxes2: torch.Tensor, mode: str = "iou", is_aligned: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    :param bboxes1:     shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
    :param bboxes2:     shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                                If ``is_aligned `` is ``True``, then m and n must be equal.
    :param mode:        Either "iou" (intersection over union) or "iof" (intersection over foreground).
    :param is_aligned:  If True, then m and n must be equal. Default False.
    :param eps:         A value added to the denominator for numerical stability. Default 1e-6.
    :return:            Tensor of shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert mode in ["iou", "iof", "giou"], "Unsupported mode {}".format(mode)
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0
    assert bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2] if bboxes1.shape[0] > 0 else 0
    cols = bboxes2.shape[-2] if bboxes2.shape[0] > 0 else 0
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return np.random.random(batch_shape + (rows,))
        else:
            return np.random.random(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = np.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = np.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = np.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = np.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = np.maximum(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = np.minimum(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = np.minimum(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = np.maximum(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps = np.array([eps])
    union = np.maximum(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clip(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = np.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def topk_(input, k, axis=1, largest=True):
    x = -input if largest else input
    if axis == 0:
        row_index = np.arange(input.shape[1 - axis])
        topk_index = np.argpartition(x, k, axis=axis)[0:k, :]
        topk_data = x[topk_index, row_index]

        topk_index_sort = np.argsort(topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:k, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(x.shape[1 - axis])[:, None]
        topk_index = np.argpartition(x, k, axis=axis)[:, 0:k]
        topk_data = x[column_index, topk_index]
        topk_data = -topk_data if largest else topk_data
        topk_index_sort = np.argsort(topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:k][column_index, topk_index_sort]

    return topk_data_sort, topk_index_sort


def compute_max_iou_anchor(ious: Tensor) -> Tensor:
    r"""
    For each anchor, find the GT with the largest IOU.

    :param ious: Tensor (float32) of shape[B, n, L], n: num_gts, L: num_anchors
    :return: is_max_iou is Tensor (float32) of shape[B, n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(dim=-2)
    is_max_iou: Tensor = torch.nn.functional.one_hot(max_iou_index, num_max_boxes).permute([0, 2, 1])
    return is_max_iou.type_as(ious)


def check_points_inside_bboxes(points: Tensor, bboxes: Tensor, center_radius_tensor: Optional[Tensor] = None, eps: float = 1e-9) -> Tensor:
    """

    :param points:                  Tensor (float32) of shape[L, 2], "xy" format, L: num_anchors
    :param bboxes:                  Tensor (float32) of shape[B, n, 4], "xmin, ymin, xmax, ymax" format
    :param center_radius_tensor:    Tensor (float32) of shape [L, 1]. Default: None.
    :param eps:                     Default: 1e-9

    :return is_in_bboxes: Tensor (float32) of shape[B, n, L], value=1. means selected
    """
    points = points.unsqueeze(0).unsqueeze(0)
    x, y = points.chunk(2, dim=-1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, dim=-1)
    # check whether `points` is in `bboxes`
    left = x - xmin
    top = y - ymin
    right = xmax - x
    bottom = ymax - y
    delta_ltrb = torch.cat([left, top, right, bottom], dim=-1)
    is_in_bboxes = delta_ltrb.min(dim=-1).values > eps
    if center_radius_tensor is not None:
        # check whether `points` is in `center_radius`
        center_radius_tensor = center_radius_tensor.unsqueeze(0).unsqueeze(0)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        left = x - (cx - center_radius_tensor)
        top = y - (cy - center_radius_tensor)
        right = (cx + center_radius_tensor) - x
        bottom = (cy + center_radius_tensor) - y
        delta_ltrb_c = torch.cat([left, top, right, bottom], dim=-1)
        is_in_center = delta_ltrb_c.min(dim=-1) > eps
        return (torch.logical_and(is_in_bboxes, is_in_center), torch.logical_or(is_in_bboxes, is_in_center))

    return is_in_bboxes.type_as(bboxes)


def gather_topk_anchors(metrics: Tensor, topk: int, largest: bool = True, topk_mask: Optional[Tensor] = None, eps: float = 1e-9) -> Tensor:
    """

    :param metrics:     Tensor(float32) of shape[B, n, L], n: num_gts, L: num_anchors
    :param topk:        The number of top elements to look for along the axis.
    :param largest:     If set to true, algorithm will sort by descending order, otherwise sort by ascending order.
    :param topk_mask:   Tensor(float32) of shape[B, n, 1], ignore bbox mask,
    :param eps:         Default: 1e-9

    :return: is_in_topk, Tensor (float32) of shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(dim=-1, keepdim=True).values > eps).type_as(metrics)
    is_in_topk = torch.nn.functional.one_hot(topk_idxs, num_anchors).sum(dim=-2).type_as(metrics)
    return is_in_topk * topk_mask


def bbox_center(boxes: Tensor) -> Tensor:
    """
    Get bbox centers from boxes.

    :param boxes:   Boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    :return:        Boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)


def compute_max_iou_gt(ious: Tensor) -> Tensor:
    """
    For each GT, find the anchor with the largest IOU.

    :param ious: Tensor (float32) of shape[B, n, L], n: num_gts, L: num_anchors
    :return:    is_max_iou, Tensor (float32) of shape[B, n, L], value=1. means selected
    """
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(dim=-1)
    is_max_iou = torch.nn.functional.one_hot(max_iou_index, num_anchors)
    return is_max_iou.astype(ious.dtype)


class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
    via Adaptive Training Sample Selection
    """

    __shared__ = ["num_classes"]

    def __init__(self, topk=9, num_classes=80, force_gt_matching=False, eps=1e-9):
        """

        :param topk: Maximum number of achors that is selected for each gt box
        :param num_classes:
        :param force_gt_matching: Guarantee that each gt box is matched to at least one anchor.
            If two gt boxes match to the same anchor, the one with the larger area will be selected.
            And the second-best achnor will be assigned to the other gt box.
        :param eps: Small constant for numerical stability
        """
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list, pad_gt_mask):
        gt2anchor_distances_list = torch.split(gt2anchor_distances, num_anchors_list, dim=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [
            0,
        ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list, num_anchors_index):
            num_anchors = distances.shape[-1]
            _, topk_idxs = torch.topk(distances, self.topk, dim=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            is_in_topk = torch.nn.functional.one_hot(topk_idxs, num_anchors).sum(dim=-2).type_as(gt2anchor_distances)
            is_in_topk_list.append(is_in_topk * pad_gt_mask)
        is_in_topk_list = torch.cat(is_in_topk_list, dim=-1)
        topk_idxs_list = torch.cat(topk_idxs_list, dim=-1)
        return is_in_topk_list, topk_idxs_list

    @torch.no_grad()
    def forward(
        self,
        anchor_bboxes: Tensor,
        num_anchors_list: list,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_gt_mask: Tensor,
        bg_index: int,
        gt_scores: Optional[Tensor] = None,
        pred_bboxes: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        This code is based on https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.

        :param anchor_bboxes:       Tensor(float32) - pre-defined anchors, shape(L, 4), "xmin, xmax, ymin, ymax" format
        :param num_anchors_list:    Number of anchors in each level
        :param gt_labels:           Tensor (int64|int32) - Label of gt_bboxes, shape(B, n, 1)
        :param gt_bboxes:           Tensor (float32) - Ground truth bboxes, shape(B, n, 4)
        :param pad_gt_mask:         Tensor (float32) - 1 means bbox, 0 means no bbox, shape(B, n, 1)
        :param bg_index:            Background index
        :param gt_scores:           Tensor (float32) - Score of gt_bboxes, shape(B, n, 1), if None, then it will initialize with one_hot label
        :param pred_bboxes:         Tensor (float32) - predicted bounding boxes, shape(B, L, 4)
        :return:
            - assigned_labels: Tensor of shape (B, L)
            - assigned_bboxes: Tensor of shape (B, L, 4)
            - assigned_scores: Tensor of shape (B, L, C), if pred_bboxes is not None, then output ious
        """
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.long, device=anchor_bboxes.device)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4], device=anchor_bboxes.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, self.num_classes], device=anchor_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        # gt2anchor_distances = (
        #     (gt_centers - anchor_centers.unsqueeze(0)).norm(2, dim=-1).reshape([batch_size, -1, num_anchors])
        # )

        gt2anchor_distances = torch.norm(gt_centers - anchor_centers.unsqueeze(0), p=2, dim=-1).reshape([batch_size, -1, num_anchors])

        # 3. on each pyramid level, selecting top-k closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk

        iou_threshold = torch.gather(iou_candidates.flatten(end_dim=-2), dim=1, index=topk_idxs.flatten(end_dim=-2))

        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(dim=-1, keepdim=True) + iou_threshold.std(dim=-1, keepdim=True)
        is_in_topk = torch.where(iou_candidates > iou_threshold, is_in_topk, torch.zeros_like(is_in_topk))

        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        # 8. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile([1, num_max_boxes, 1])
            mask_positive = torch.where(mask_max_iou, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        # assigned target
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(gt_labels.flatten(), index=assigned_gt_index.flatten(), dim=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))

        # assigned_bboxes = torch.gather(gt_bboxes.reshape([-1, 4]), index=assigned_gt_index.flatten(), dim=0)
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_index.flatten(), :]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = torch.nn.functional.one_hot(assigned_labels, self.num_classes + 1).float()
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(assigned_scores, index=torch.tensor(ind, device=assigned_scores.device), dim=-1)
        if pred_bboxes is not None:
            # assigned iou
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(dim=-2).values.unsqueeze(-1)
            assigned_scores *= ious
        elif gt_scores is not None:
            gather_scores = torch.gather(gt_scores.flatten(), assigned_gt_index.flatten(), dim=0)
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = torch.where(mask_positive_sum > 0, gather_scores, torch.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)

        return assigned_labels, assigned_bboxes, assigned_scores


class TaskAlignedAssigner(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection"""

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        """

        :param topk: Maximum number of achors that is selected for each gt box
        :param alpha: Power factor for class probabilities of predicted boxes (Used compute alignment metric)
        :param beta: Power factor for IoU score of predicted boxes (Used compute alignment metric)
        :param eps: Small constant for numerical stability
        """
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        num_anchors_list: list,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_gt_mask: Tensor,
        bg_index: int,
        gt_scores: Optional[Tensor] = None,
    ):
        """
        This code is based on https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.

        :param pred_scores: Tensor (float32): predicted class probability, shape(B, L, C)
        :param pred_bboxes: Tensor (float32): predicted bounding boxes, shape(B, L, 4)
        :param anchor_points: Tensor (float32): pre-defined anchors, shape(L, 2), "cxcy" format
        :param num_anchors_list: List ( num of anchors in each level, shape(L)
        :param gt_labels: Tensor (int64|int32): Label of gt_bboxes, shape(B, n, 1)
        :param gt_bboxes: Tensor (float32): Ground truth bboxes, shape(B, n, 4)
        :param pad_gt_mask: Tensor (float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        :param bg_index: int ( background index
        :param gt_scores: Tensor (one, float32) Score of gt_bboxes, shape(B, n, 1)
        :return:
            - assigned_labels, Tensor of shape (B, L)
            - assigned_bboxes, Tensor of shape (B, L, 4)
            - assigned_scores, Tensor of shape (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.long, device=gt_labels.device)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4], device=gt_labels.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=gt_labels.device)
            return assigned_labels, assigned_bboxes, assigned_scores

        # compute iou between gt and pred bbox, [B, n, L]
        ious = batch_iou_similarity(gt_bboxes, pred_bboxes)
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

        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_index.flatten(), :]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

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

        return assigned_labels, assigned_bboxes, assigned_scores


class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630

    :param loss_weight: giou loss weight, default as 1
    :param eps:         epsilon to avoid divide by zero, default as 1e-10
    :param reduction:   Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-10, reduction: str = "none"):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def bbox_overlap(self, box1: Tensor, box2: Tensor, eps: float = 1e-10) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the iou of box1 and box2.

        :param box1:    box1 with the shape (..., 4)
        :param box2:    box1 with the shape (..., 4)
        :param eps:     epsilon to avoid divide by zero
        :return:
            - iou:      iou of box1 and box2
            - overlap:  overlap of box1 and box2
            - union:    union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clip(0)
        h_inter = (ykis2 - ykis1).clip(0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox: Tensor, gbox: Tensor, iou_weight=1.0, loc_reweight=None):
        # x1, y1, x2, y2 = torch.split(pbox, split_size_or_sections=4, dim=-1)
        # x1g, y1g, x2g, y2g = torch.split(gbox, split_size_or_sections=4, dim=-1)

        x1, y1, x2, y2 = pbox.chunk(4, dim=-1)
        x1g, y1g, x2g, y2g = gbox.chunk(4, dim=-1)

        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == "none":
            loss = giou
        elif self.reduction == "sum":
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight


@register_loss(Losses.PPYOLOE_LOSS)
class PPYoloELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_varifocal_loss: bool = True,
        use_static_assigner: bool = True,
        reg_max: int = 16,
        classification_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        dfl_loss_weight: float = 0.5,
    ):
        """
        :param num_classes: Number of classes
        :param use_varifocal_loss: Whether to use Varifocal loss for classification loss; otherwise use Focal loss
        :param static_assigner_epoch: Whether to use static assigner or Task-Aligned assigner
        :param classification_loss_weight: Classification loss weight
        :param iou_loss_weight: IoU loss weight
        :param dfl_loss_weight: DFL loss weight
        :param reg_max: Number of regression bins (Must match the number of bins in the PPYoloE head)
        """
        super().__init__()
        self.use_varifocal_loss = use_varifocal_loss
        self.classification_loss_weight = classification_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.iou_loss_weight = iou_loss_weight

        self.iou_loss = GIoULoss()
        self.static_assigner = ATSSAssigner(topk=9, num_classes=num_classes)
        self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)
        self.use_static_assigner = use_static_assigner
        self.reg_max = reg_max
        self.num_classes = num_classes

        # Same as in PPYoloE head
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj)

    @torch.no_grad()
    def _yolox_targets_to_ppyolo(self, targets: torch.Tensor, batch_size: int) -> Mapping[str, torch.Tensor]:
        """
        Convert targets from YoloX format to PPYolo since its the easiest (not the cleanest) way to
        have PP Yolo training & metrics computed

        :param targets: (N, 6) format of bboxes is meant to be LABEL_CXCYWH (index, c, cx, cy, w, h)
        :return: (Dictionary [str,Tensor]) with keys:
         - gt_class: (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
         - gt_bbox: (Tensor, float32): Ground truth bboxes, shape(B, n, 4) in x1y1x2y2 format
         - pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        """
        image_index = targets[:, 0]
        gt_class = targets[:, 1:2].long()
        gt_bbox = cxcywh_to_xyxy(targets[:, 2:6], image_shape=None)

        per_image_class = []
        per_image_bbox = []
        per_image_pad_mask = []

        max_boxes = 0
        for i in range(batch_size):
            mask = image_index == i

            image_labels = gt_class[mask]
            image_bboxes = gt_bbox[mask, :]
            valid_bboxes = image_bboxes.sum(dim=1, keepdims=True) > 0

            per_image_class.append(image_labels)
            per_image_bbox.append(image_bboxes)
            per_image_pad_mask.append(valid_bboxes)

            max_boxes = max(max_boxes, mask.sum().item())

        for i in range(batch_size):
            elements_to_pad = max_boxes - len(per_image_class[i])
            padding_left = 0
            padding_right = 0
            padding_top = 0
            padding_bottom = elements_to_pad
            pad = padding_left, padding_right, padding_top, padding_bottom
            per_image_class[i] = F.pad(per_image_class[i], pad, mode="constant", value=0)
            per_image_bbox[i] = F.pad(per_image_bbox[i], pad, mode="constant", value=0)
            per_image_pad_mask[i] = F.pad(per_image_pad_mask[i], pad, mode="constant", value=0)

        return {
            "gt_class": torch.stack(per_image_class, dim=0),
            "gt_bbox": torch.stack(per_image_bbox, dim=0),
            "pad_gt_mask": torch.stack(per_image_pad_mask, dim=0),
        }

    def forward(
        self,
        outputs: Union[
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
        ],
        targets: Tensor,
    ) -> Mapping[str, Tensor]:
        """
        :param outputs: Tuple of pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor
        :param targets: (Dictionary [str,Tensor]) with keys:
         - gt_class: (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
         - gt_bbox: (Tensor, float32): Ground truth bboxes, shape(B, n, 4) in x1y1x2y2 format
         - pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        :return:
        """
        # in test/eval mode the model outputs a tuple where the second item is the raw predictions
        if isinstance(outputs, tuple) and len(outputs) == 2:
            # in test/eval mode the Yolo model outputs a tuple where the second item is the raw predictions
            _, predictions = outputs
        else:
            predictions = outputs

        (
            pred_scores,
            pred_distri,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = predictions

        targets = self._yolox_targets_to_ppyolo(targets, batch_size=pred_scores.size(0))  # yolox -> ppyolo

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = targets["gt_class"]
        gt_bboxes = targets["gt_bbox"]
        pad_gt_mask = targets["pad_gt_mask"]
        # label assignment
        if self.use_static_assigner:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchor_bboxes=anchors,
                num_anchors_list=num_anchors_list,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                pad_gt_mask=pad_gt_mask,
                bg_index=self.num_classes,
                pred_bboxes=pred_bboxes.detach() * stride_tensor,
            )
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores=pred_scores.detach().sigmoid(),  # Pred scores are logits on training for numerical stability
                pred_bboxes=pred_bboxes.detach() * stride_tensor,
                anchor_points=anchor_points,
                num_anchors_list=num_anchors_list,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                pad_gt_mask=pad_gt_mask,
                bg_index=self.num_classes,
            )
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = torch.nn.functional.one_hot(assigned_labels, self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores, one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if super_gradients.is_distributed():
            torch.distributed.all_reduce(assigned_scores_sum, op=torch.distributed.ReduceOp.SUM)
            assigned_scores_sum /= get_world_size()
        assigned_scores_sum = torch.clip(assigned_scores_sum, min=1.0)
        loss_cls /= assigned_scores_sum

        loss_iou, loss_dfl = self._bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points_s,
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            assigned_scores_sum,
        )

        loss = self.classification_loss_weight * loss_cls + self.iou_loss_weight * loss_iou + self.dfl_loss_weight * loss_dfl
        log_losses = torch.stack([loss_cls.detach(), loss_iou.detach(), loss_dfl.detach(), loss.detach()])

        return loss, log_losses

    @property
    def component_names(self):
        return ["loss_cls", "loss_iou", "loss_dfl", "loss"]

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

    def _bbox_loss(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        assigned_labels,
        assigned_bboxes,
        assigned_scores,
        assigned_scores_sum,
    ):
        # select positive samples mask
        mask_positive = assigned_labels != self.num_classes
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_iou = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).tile([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_iou = torch.zeros([], device=pred_bboxes.device)
            loss_dfl = pred_dist.sum() * 0.0
        return loss_iou, loss_dfl

    def _bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor):
        b, l, *_ = pred_dist.size()
        pred_dist = torch.softmax(pred_dist.reshape([b, l, 4, self.reg_max + 1]), dim=-1)
        pred_dist = torch.nn.functional.conv2d(pred_dist.permute(0, 3, 1, 2), self.proj_conv).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], dim=-1).clip(0, self.reg_max - 0.01)

    @staticmethod
    def _focal_loss(pred_logits: Tensor, label: Tensor, alpha=0.25, gamma=2.0) -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = (pred_score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = -weight * (label * torch.nn.functional.logsigmoid(pred_logits) + (1 - label) * torch.nn.functional.logsigmoid(-pred_logits))
        return loss.sum()

    @staticmethod
    def _varifocal_loss(pred_logits: Tensor, gt_score: Tensor, label: Tensor, alpha=0.75, gamma=2.0) -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = -weight * (gt_score * torch.nn.functional.logsigmoid(pred_logits) + (1 - gt_score) * torch.nn.functional.logsigmoid(-pred_logits))
        return loss.sum()
