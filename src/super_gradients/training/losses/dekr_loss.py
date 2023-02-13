from typing import Tuple

import torch
from torch import Tensor, nn


class DEKRLoss(nn.Module):
    """
    Implementation of the loss function from the "Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression"
    paper (https://arxiv.org/abs/2104.02300)

    This loss should be used in conjunction with DEKRTargetsGenerator.
    """

    def __init__(self, heatmap_loss_factor: float = 1.0, offset_loss_factor: float = 0.1):
        """

        :param heatmap_loss_factor: Weighting factor for heatmap loss
        :param offset_loss_factor: Weighting factor for offset loss
        """
        super().__init__()
        self.heatmap_loss_factor = float(heatmap_loss_factor)
        self.offset_loss_factor = float(offset_loss_factor)

    @property
    def component_names(self):
        """
        Names of individual loss components for logging during training.
        """
        return ["heatmap", "offset", "total"]

    def forward(self, predictions: Tuple[Tensor, Tensor], targets: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """

        :param predictions: Tuple of (heatmap, offset) predictions.
            heatmap is of shape (B, NumJoints + 1, H, W)
            offset is of shape (B, NumJoints * 2, H, W)

        :param targets: Tuple of (heatmap, mask, offset, offset_weight).
            heatmap is of shape (B, NumJoints + 1, H, W)
            mask is of shape (B, NumJoints + 1, H, W)
            offset is of shape (B, NumJoints * 2, H, W)
            offset_weight is of shape (B, NumJoints * 2, H, W)

        :return: Tuple of (loss, loss_components)
            loss is a scalar tensor with the total loss
            loss_components is a tensor of shape (3,) containing the individual loss components for logging (detached from the graph)
        """
        pred_heatmap, pred_offset = predictions
        gt_heatmap, mask, gt_offset, offset_weight = targets

        heatmap_loss = self.heatmap_loss(pred_heatmap, gt_heatmap, mask) * self.heatmap_loss_factor
        offset_loss = self.offset_loss(pred_offset, gt_offset, offset_weight) * self.offset_loss_factor

        loss = heatmap_loss + offset_loss
        components = torch.cat(
            (
                heatmap_loss.unsqueeze(0),
                offset_loss.unsqueeze(0),
                loss.unsqueeze(0),
            )
        ).detach()

        return loss, components

    def heatmap_loss(self, pred_heatmap, true_heatmap, mask):
        loss = ((pred_heatmap - true_heatmap) ** 2) * mask
        loss = loss.mean()
        return loss

    def offset_loss(self, pred_offsets, true_offsets, weights):
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred_offsets, true_offsets) * weights
        if num_pos == 0:
            num_pos = 1.0
        loss = loss.sum() / num_pos
        return loss

    def smooth_l1_loss(self, pred, gt, beta=1.0 / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5 * l1_loss**2 / beta, l1_loss - 0.5 * beta)
        return loss
