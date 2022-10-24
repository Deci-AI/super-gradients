from typing import Mapping
from typing import Tuple

import torch
from torch import nn, Tensor

import super_gradients
from super_gradients.training.models.detection_models.pp_yolo_e.assigner import (
    ATSSAssigner,
    TaskAlignedAssigner,
    batch_distance2bbox,
)
from super_gradients.training.utils.callbacks import TrainingStageSwitchCallbackBase, PhaseContext
from super_gradients.training.utils.distributed_training_utils import (
    get_world_size,
)


class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1.0, eps=1e-10, reduction="none"):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
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


class PPYoloLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_varifocal_loss: bool = True,
        use_static_assigner: bool = True,
        use_l1_loss: bool = False,
        loss_weight={
            "class": 1.0,
            "iou": 2.5,
            "dfl": 0.5,
        },
        reg_max: int = 16,
    ):
        """

        :param use_varifocal_loss:
        :param loss_weight:
        :param static_assigner_epoch:
        """
        super().__init__()
        self.use_varifocal_loss = use_varifocal_loss
        self.loss_weight = loss_weight
        self.iou_loss = GIoULoss()
        self.static_assigner = ATSSAssigner(topk=9)
        self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)
        self.use_static_assigner = use_static_assigner
        self.use_l1_loss = use_l1_loss
        self.reg_max = reg_max
        self.num_classes = num_classes

        # Same as in PPYoloE head
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj)

    def forward(
        self,
        outputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        targets: Mapping[str, Tensor],
    ) -> Mapping[str, Tensor]:
        """
        :param outputs: Tuple of pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor
        :param targets: (Dictionary [str,Tensor]) with keys:
         - gt_class: (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
         - gt_bbox: (Tensor, float32): Ground truth bboxes, shape(B, n, 4) in x1y1x2y2 format
         - pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        :return:
        """
        (
            pred_scores,
            pred_distri,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = outputs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = targets["gt_class"]
        gt_bboxes = targets["gt_bbox"]
        pad_gt_mask = targets["pad_gt_mask"]
        # label assignment
        if self.use_static_assigner:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                pad_gt_mask=pad_gt_mask,
                bg_index=self.num_classes,
                pred_bboxes=pred_bboxes.detach() * stride_tensor,
            )
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
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

        loss_l1, loss_iou, loss_dfl = self._bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points_s,
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            assigned_scores_sum,
        )

        loss = (
            self.loss_weight["class"] * loss_cls
            + self.loss_weight["iou"] * loss_iou
            + self.loss_weight["dfl"] * loss_dfl
        )

        # This is not present in the original PPYoloE loss, but may be beneficial to enable after some time
        if self.use_l1_loss:
            loss += self.loss_weight["l1"] * loss_l1

        out_dict = {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_iou": loss_iou,
            "loss_dfl": loss_dfl,
            "loss_l1": loss_l1,
        }

        return out_dict

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

            loss_l1 = torch.nn.functional.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).tile([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([1], device=pred_bboxes.device)
            loss_iou = torch.zeros([1], device=pred_bboxes.device)
            loss_dfl = pred_dist.sum() * 0.0
        return loss_l1, loss_iou, loss_dfl

    def _bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor):
        b, l, *_ = pred_dist.size()
        pred_dist = torch.softmax(pred_dist.reshape([b, l, 4, self.reg_max + 1]), dim=-1)
        pred_dist = torch.nn.functional.conv2d(pred_dist.permute(0, 3, 1, 2), self.proj_conv).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.concat([lt, rb], dim=-1).clip(0, self.reg_max - 0.01)

    @staticmethod
    def _focal_loss(score: Tensor, label: Tensor, alpha=0.25, gamma=2.0) -> Tensor:
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        # loss = torch.nn.functional.binary_cross_entropy(score, label, weight=weight, reduction="sum")
        # The function 'binary_cross_entropy' is not differentiable with respect to argument 'weight'.
        # loss = torch.nn.functional.binary_cross_entropy(pred_score, gt_score, weight=weight, reduction="sum")
        # TODO: Can improve loss by using logsigmoid with has better numerical stability
        loss = -weight * (label * score.log() + (1 - label) * (1 - score).log())
        return loss.sum()

    @staticmethod
    def _varifocal_loss(pred_score: Tensor, gt_score: Tensor, label: Tensor, alpha=0.75, gamma=2.0) -> Tensor:
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        # The function 'binary_cross_entropy' is not differentiable with respect to argument 'weight'.
        # loss = torch.nn.functional.binary_cross_entropy(pred_score, gt_score, weight=weight, reduction="sum")
        # TODO: Can improve loss by using logsigmoid with has better numerical stability
        loss = -weight * (gt_score * pred_score.log() + (1 - gt_score) * (1 - pred_score).log())
        return loss.sum()


class PPYoloTrainingStageSwitchCallback(TrainingStageSwitchCallbackBase):
    """
    PPYoloTrainingStageSwitchCallback

    Training stage switch for PPYolo training.
    It changes static bbox assigner to a task aligned assigned after certain number of epochs passed

    """

    def __init__(
        self,
        static_assigner_end_epoch: int = 30,
    ):
        super().__init__(next_stage_start_epoch=static_assigner_end_epoch)

    def apply_stage_change(self, context: PhaseContext):
        assert isinstance(context.criterion, PPYoloLoss)
        context.criterion.use_static_assigner = False
