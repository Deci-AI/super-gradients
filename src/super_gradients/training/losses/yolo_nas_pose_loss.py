from typing import Mapping, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import super_gradients
from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from super_gradients.training.utils.distributed_training_utils import (
    get_world_size,
)
from .ppyolo_loss import GIoULoss, TaskAlignedAssigner, BoxesAssignmentResult


@register_loss(Losses.YOLONAS_POSE_LOSS)
class YoloNASPoseLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        oks_sigmas: Union[List[float], np.ndarray, Tensor],
        use_varifocal_loss: bool = False,
        reg_max: int = 16,
        classification_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        dfl_loss_weight: float = 0.5,
        pose_cls_loss_weight: float = 1.0,
        pose_reg_loss_weight: float = 1.0,
    ):
        """
        :param num_classes: Number of keypoints
        :param use_varifocal_loss: Whether to use Varifocal loss for classification loss; otherwise use Focal loss
        :param classification_loss_weight: Classification loss weight
        :param iou_loss_weight: IoU loss weight
        :param dfl_loss_weight: DFL loss weight
        :param reg_max: Number of regression bins (Must match the number of bins in the PPYoloE head)
        :param pose_cls_loss_weight: Pose classification loss weight
        :param pose_reg_loss_weight: Pose regression loss weight
        """
        super().__init__()
        self.use_varifocal_loss = use_varifocal_loss
        self.classification_loss_weight = classification_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.iou_loss_weight = iou_loss_weight

        self.iou_loss = GIoULoss()
        self.reg_max = reg_max
        self.num_keypoints = num_classes
        self.num_classes = 1  # We have only one class (person)
        self.oks_sigmas = torch.tensor(oks_sigmas)
        self.pose_cls_loss_weight = pose_cls_loss_weight
        self.pose_reg_loss_weight = pose_reg_loss_weight
        self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)

        # Same as in PPYoloE head
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj)

    @torch.no_grad()
    def _convert_yolo_nas_pose_targets_to_ppyolo(self, targets: Tuple[Tensor, Tensor], batch_size: int) -> Mapping[str, torch.Tensor]:
        """
        Convert targets to PPYoloE-compatible format since it's the easiest (not the cleanest) way to
        have PP Yolo training & metrics computed

        :param targets: Tuple (boxes, joints, mask)
            - boxes: [N, 5] (batch_index, x1, y1, x2, y2)
            - joints: [N, num_joints, 4] (batch_index, x, y, visibility)
        :return: (Dictionary [str,Tensor]) with keys:
         - gt_class: (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
         - gt_bbox: (Tensor, float32): Ground truth bboxes, shape(B, n, 4) in x1y1x2y2 format
         - pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        """
        target_boxes, target_joints = targets

        image_index = target_boxes[:, 0]
        gt_bbox = target_boxes[:, 1:5]

        per_image_class = []
        per_image_bbox = []
        per_image_pad_mask = []

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

        new_targets = {
            "gt_class": torch.stack(per_image_class, dim=0),
            "gt_bbox": torch.stack(per_image_bbox, dim=0),
            "pad_gt_mask": torch.stack(per_image_pad_mask, dim=0),
            "gt_poses": target_joints,
        }
        return new_targets

    def forward(
        self,
        outputs: Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]],
        targets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        :param outputs: Tuple of pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor
        :param targets: A single tensor of shape (N, 1 + 4 + Num Joints * 3) (batch_index, x1, y1, x2, y2, [x, y, visibility] * Num Joints)
        :return:
        """
        _, predictions = outputs

        (
            pred_scores,
            pred_distri,
            pred_pose_logits,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = predictions

        targets = self._convert_yolo_nas_pose_targets_to_ppyolo(targets, batch_size=pred_scores.size(0))  # targets -> ppyolo

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = targets["gt_class"]
        gt_bboxes = targets["gt_bbox"]
        pad_gt_mask = targets["pad_gt_mask"]

        # label assignment
        assign_result = self.assigner(
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

        assigned_labels = assign_result.assigned_labels
        assigned_bboxes = assign_result.assigned_bboxes
        assigned_scores = assign_result.assigned_scores

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

        loss_pose_cls, loss_pose_reg = self._pose_loss(
            pred_pose_logits,
            true_keypoints=targets["gt_poses"],
            stride_tensor=stride_tensor,
            anchor_points=anchor_points_s,
            assign_result=assign_result,
            assigned_scores_sum=assigned_scores_sum,
        )

        loss = (
            self.classification_loss_weight * loss_cls
            + self.iou_loss_weight * loss_iou
            + self.dfl_loss_weight * loss_dfl
            + self.pose_cls_loss_weight * loss_pose_cls
            + self.pose_reg_loss_weight * loss_pose_reg
        )
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

    def _keypoint_loss(self, pred_kpts: Tensor, gt_kpts: Tensor, area: Tensor, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        """

        :param pred_kpts: [Num Instances, Num Joints, 3] - (x, y, visibility)
        :param gt_kpts: [Num Instances, Num Joints, 3] - (x, y, visibility)
        :param sigmas: [Num Joints] - Sigma for each joint
        :param area: [Num Instances, 1] - Area of the corresponding bounding box
        :return: Tuple of regression loss and classification loss
        """
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        kpt_mask = gt_kpts[..., 2] != 0
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        regression_loss = kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()
        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_kpts[..., 2], kpt_mask.float(), reduction="mean")
        return regression_loss, classification_loss

    def _pose_loss(self, pose_regression_list, true_keypoints, anchor_points, stride_tensor, assign_result: BoxesAssignmentResult, assigned_scores_sum):
        """

        :param pose_regression_list: [B, Anchors, C, 3]
        :param true_keypoints: [N, Num Joints, 4] - (batch_index, x, y, visibility)
        :param anchor_points:
        :param assign_result:
        :param assigned_scores_sum:
        :return:
        """

        assigned_labels = assign_result.assigned_labels
        mask_positive = assigned_labels != self.num_classes
        batch_size = pose_regression_list.size(0)
        loss_pose_cls = torch.zeros([], device=pose_regression_list.device)
        loss_pose_reg = torch.zeros([], device=pose_regression_list.device)

        batch_index = true_keypoints[:, 0, 0].long()
        true_keypoints = true_keypoints[:, :, 1:]  # Remove batch index

        # Compute absolute coordinates of keypoints in image space
        pred_pose_coords = pose_regression_list + 0
        pred_pose_coords[:, :, :, 0:2] += anchor_points.unsqueeze(0).unsqueeze(2)
        pred_pose_coords[:, :, :, 0:2] *= stride_tensor.unsqueeze(0).unsqueeze(2)

        # pred_pose_coords = (pose_regression_list[:, :, : ,0:2] + anchor_points.unsqueeze(0).unsqueeze(2)) * stride_tensor.unsqueeze(0).unsqueeze(2)

        # Assigned boxes area divided by stride so to compute area correctly we need to multiply by stride back
        assigned_bboxes_image_space = assign_result.assigned_bboxes * stride_tensor.unsqueeze(0)

        # pos/neg loss
        for i in range(batch_size):
            if mask_positive[i].sum():
                image_level_mask = mask_positive[i]
                idx = assign_result.assigned_gt_index_non_flat[i][image_level_mask]
                gt_kpt = true_keypoints[batch_index == i][idx]
                # gt_kpt[..., 0:1] /= stride_tensor[image_level_mask].unsqueeze(1)
                # gt_kpt[..., 1:2] /= stride_tensor[image_level_mask].unsqueeze(1)
                area = self._xyxy_box_area(assigned_bboxes_image_space[i][image_level_mask])
                pred_kpt = pred_pose_coords[i][image_level_mask]
                loss = self._keypoint_loss(pred_kpt, gt_kpt, area=area, sigmas=self.oks_sigmas.to(pred_kpt.device))  # pose loss
                loss_pose_cls += loss[0]
                loss_pose_reg += loss[1]

        return loss_pose_cls, loss_pose_reg

    def _xyxy_box_area(self, boxes):
        return (boxes[:, 2:3] - boxes[:, 0:1]) * (boxes[:, 3:4] - boxes[:, 1:2])

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
