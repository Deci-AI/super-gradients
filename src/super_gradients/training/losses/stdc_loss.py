from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.training.utils.segmentation_utils import to_one_hot
from super_gradients.training.losses.ohem_ce_loss import OhemCELoss, OhemBCELoss, OhemLoss
from super_gradients.training.losses.dice_loss import BinaryDiceLoss


class DetailAggregateModule(nn.Module):
    """
    DetailAggregateModule to create ground-truth spatial details map. Given ground-truth segmentation masks and using
     laplacian kernels this module create feature-maps with special attention to classes edges aka details.
    """

    _LAPLACIAN_KERNEL = [-1, -1, -1, -1, 8, -1, -1, -1, -1]
    _INITIAL_FUSE_KERNEL = [[6.0 / 10], [3.0 / 10], [1.0 / 10]]

    def __init__(self, num_classes: int, ignore_label: int, detail_threshold: float = 1.0, learnable_fusing_kernel: bool = True):
        """
        :param detail_threshold: threshold to define a pixel as edge after laplacian. must be a value between 1 and 8,
            lower value for smooth edges, high value for fine edges.
        :param learnable_fusing_kernel: whether the 1x1 conv map of strided maps is learnable or not.
        """
        super().__init__()
        assert 1 <= detail_threshold <= 8, f"Detail threshold must be a value between 1 and 8, found: {detail_threshold}"

        self.device = None
        self.detail_threshold = detail_threshold
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        # laplacian dw-convolution, each channel is a class label. apply laplacian filter once for each channel.
        self.laplacian_kernel = torch.tensor(self._LAPLACIAN_KERNEL, dtype=torch.float32).reshape(1, 1, 3, 3).expand(num_classes, 1, 3, 3).requires_grad_(False)
        # init param for 1x1 conv of strided gaussian feature maps.
        self.fuse_kernel = torch.tensor(self._INITIAL_FUSE_KERNEL, dtype=torch.float32).reshape(1, 3, 1, 1).requires_grad_(learnable_fusing_kernel)
        if learnable_fusing_kernel:
            self.fuse_kernel = torch.nn.Parameter(self.fuse_kernel)

    def forward(self, gt_masks: torch.Tensor):
        if self.device is None:
            self._set_kernels_to_device(gt_masks.device)
        if self.num_classes > 1:
            one_hot = to_one_hot(gt_masks, self.num_classes, self.ignore_label).float()
        else:
            one_hot = gt_masks.unsqueeze(1).float()
        # create binary detail maps using filters withs strides of 1, 2 and 4.
        boundary_targets = F.conv2d(one_hot, self.laplacian_kernel, stride=1, padding=1, groups=self.num_classes)
        boundary_targets_x2 = F.conv2d(one_hot, self.laplacian_kernel, stride=2, padding=1, groups=self.num_classes)
        boundary_targets_x4 = F.conv2d(one_hot, self.laplacian_kernel, stride=4, padding=1, groups=self.num_classes)

        boundary_targets = self._to_one_channel_binary(boundary_targets, self.detail_threshold)
        boundary_targets_x2 = self._to_one_channel_binary(boundary_targets_x2, self.detail_threshold)
        boundary_targets_x4 = self._to_one_channel_binary(boundary_targets_x4, self.detail_threshold)

        boundary_targets_x4 = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode="nearest")
        boundary_targets_x2 = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode="nearest")

        boundary_targets = torch.cat((boundary_targets, boundary_targets_x2, boundary_targets_x4), dim=1)

        boundary_targets = F.conv2d(boundary_targets, self.fuse_kernel)
        boundary_targets = self._to_one_channel_binary(boundary_targets, 0.3)

        return boundary_targets

    def _set_kernels_to_device(self, device: str):
        self.device = device
        self.laplacian_kernel = self.laplacian_kernel.to(device)
        self.fuse_kernel = self.fuse_kernel.to(device)

    @staticmethod
    def _to_one_channel_binary(x: torch.Tensor, threshold: float):
        """
        Flatten channels, and turn to binary tensor. if at least one pixel class is above threshold, flatten value is 1,
        'or' operator.
        """
        x = x.max(dim=1, keepdim=True)[0]
        x[x < threshold] = 0
        x[x >= threshold] = 1
        return x


class DetailLoss(_Loss):
    """
    STDC DetailLoss applied on  details features from higher resolution and ground-truth details map.
    Loss combination of BCE loss and BinaryDice loss
    """

    def __init__(self, weights: list = [1.0, 1.0]):
        """
        :param weights: weight to apply for each part of the loss contributions, [BCE, Dice] respectively.
        """
        super().__init__()
        assert len(weights) == 2, f"Only 2 weight elements are required for BCE-Dice loss combo, found: {len(weights)}"
        self.weights = weights
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss(apply_sigmoid=True)

    def forward(self, detail_out: torch.Tensor, detail_target: torch.Tensor):
        """
        :param detail_out: predicted detail map.
        :param detail_target: ground-truth detail loss, output of DetailAggregateModule.
        """
        bce_loss = self.bce_with_logits(detail_out, detail_target)
        dice_loss = self.dice_loss(detail_out, detail_target)
        return self.weights[0] * bce_loss + self.weights[1] * dice_loss


@register_loss(Losses.STDC_LOSS)
class STDCLoss(_Loss):
    """
    Loss class of STDC-Seg training.
    """

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.7,
        num_aux_heads: int = 2,
        num_detail_heads: int = 1,
        weights: Union[tuple, list] = (1, 1, 1, 1),
        detail_weights: Union[tuple, list] = (1, 1),
        mining_percent: float = 0.1,
        detail_threshold: float = 1.0,
        learnable_fusing_kernel: bool = True,
        ignore_index: int = None,
        ohem_criteria: OhemLoss = None,
    ):
        """
        :param threshold: Online hard-mining probability threshold.
        :param num_aux_heads: num of auxiliary heads.
        :param num_detail_heads: num of detail heads.
        :param weights: Loss lambda weights.
        :param detail_weights: weights for (Dice, BCE) losses parts in DetailLoss.
        :param mining_percent: mining percentage.
        :param detail_threshold: detail threshold to create binary details features in DetailLoss.
        :param learnable_fusing_kernel: whether DetailAggregateModule params are learnable or not.
        :param ohem_criteria: OhemLoss criterion component of STDC. When none is given, it will be derrived according
         to num_classes (i.e OhemCELoss if num_classes > 1 and OhemBCELoss otherwise).
        """
        super().__init__()

        assert len(weights) == num_aux_heads + num_detail_heads + 1, "Lambda loss weights must be in same size as loss items."

        self.weights = weights
        self.use_detail = num_detail_heads > 0

        self.num_aux_heads = num_aux_heads
        self.num_detail_heads = num_detail_heads

        if self.use_detail:
            self.detail_module = DetailAggregateModule(
                num_classes=num_classes, detail_threshold=detail_threshold, ignore_label=ignore_index, learnable_fusing_kernel=learnable_fusing_kernel
            )
            self.detail_loss = DetailLoss(weights=detail_weights)

        if ohem_criteria is None:
            if num_classes > 1:
                ohem_criteria = OhemCELoss(threshold=threshold, mining_percent=mining_percent, ignore_lb=ignore_index)
            else:
                ohem_criteria = OhemBCELoss(threshold=threshold, mining_percent=mining_percent)

        self.ce_ohem = ohem_criteria
        self.num_classes = num_classes

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return ["main_loss", "aux_loss1", "aux_loss2", "detail_loss", "loss"]

    def forward(self, preds: Tuple[torch.Tensor], target: torch.Tensor):
        """
        :param preds: Model output predictions, must be in the followed format:
         [Main-feats, Aux-feats[0], ..., Aux-feats[num_auxs-1], Detail-feats[0], ..., Detail-feats[num_details-1]
        """
        assert (
            len(preds) == self.num_aux_heads + self.num_detail_heads + 1
        ), f"Wrong num of predictions tensors for STDC loss, expected {self.num_aux_heads + self.num_detail_heads + 1} found {len(preds)}"
        losses = []
        total_loss = 0

        # classification and auxiliary loss
        for i in range(0, 1 + self.num_aux_heads):
            ce_loss = self.ce_ohem(preds[i], target)
            total_loss += ce_loss * self.weights[i]
            losses.append(ce_loss)

        # detail heads loss
        if self.use_detail:
            gt_binary_mask = self.detail_module(target)
            for i in range(1 + self.num_aux_heads, len(preds)):
                detail_loss = self.detail_loss(preds[i], gt_binary_mask)
                total_loss += self.weights[i] * detail_loss
                losses.append(detail_loss)

        losses.append(total_loss)

        return total_loss, torch.stack(losses, dim=0).detach()

    def get_train_named_params(self):
        """
        Expose DetailAggregateModule learnable parameters to be passed to the optimizer.
        """
        if self.use_detail:
            return list(self.detail_module.named_parameters())
