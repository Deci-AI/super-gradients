import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing import Union, Tuple

from super_gradients.training.losses.dice_loss import DiceLoss, BinaryDiceLoss
from super_gradients.training.utils.segmentation_utils import target_to_binary_edge

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.training.losses.mask_loss import MaskAttentionLoss


@register_loss(Losses.DICE_CE_EDGE_LOSS)
class DiceCEEdgeLoss(_Loss):
    def __init__(
        self,
        num_classes: int,
        num_aux_heads: int = 2,
        num_detail_heads: int = 1,
        weights: Union[tuple, list] = (1, 1, 1, 1),
        dice_ce_weights: Union[tuple, list] = (1, 1),
        ignore_index: int = -100,
        edge_kernel: int = 3,
        ce_edge_weights: Union[tuple, list] = (0.5, 0.5),
    ):
        """
        Total loss is computed as follows:

            Loss-cls-edge = λ1 * CE + λ2 * M * CE , where [λ1, λ2] are ce_edge_weights.

        For each Main feature maps and auxiliary heads the loss is calculated as:

            Loss-main-aux = λ3 * Loss-cls-edge + λ4 * Loss-Dice, where [λ3, λ4] are dice_ce_weights.

        For Feature maps defined as detail maps that predicts only the edge mask, the loss is computed as follow:

            Loss-detail = BinaryCE + BinaryDice

        Finally the total loss is computed as follows for the whole feature maps:

            Loss = Σw[i] * Loss-main-aux[i] + Σw[j] * Loss-detail[j], where `w` is defined as the `weights` argument
                `i` in [0, 1 + num_aux_heads], 1 is for the main feature map.
                `j` in [1 + num_aux_heads, 1 + num_aux_heads + num_detail_heads].


        :param num_aux_heads: num of auxiliary heads.
        :param num_detail_heads: num of detail heads.
        :param weights: Loss lambda weights.
        :param dice_ce_weights: weights lambdas between (Dice, CE) losses.
        :param edge_kernel: kernel size of dilation erosion convolutions for creating the edge feature map.
        :param ce_edge_weights: weights lambdas between regular CE and edge attention CE.
        """
        super().__init__()
        # Check that arguments are valid.
        assert len(weights) == num_aux_heads + num_detail_heads + 1, "Lambda loss weights must be in same size as loss items."
        assert len(dice_ce_weights) == 2, f"dice_ce_weights must an iterable with size 2, found: {len(dice_ce_weights)}"
        assert len(ce_edge_weights) == 2, f"dice_ce_weights must an iterable with size 2, found: {len(ce_edge_weights)}"

        self.edge_kernel = edge_kernel
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.weights = weights
        self.dice_ce_weights = dice_ce_weights
        self.use_detail = num_detail_heads > 0

        self.num_aux_heads = num_aux_heads
        self.num_detail_heads = num_detail_heads

        if self.use_detail:
            self.bce = nn.BCEWithLogitsLoss()
            self.binary_dice = BinaryDiceLoss(apply_sigmoid=True)

        self.ce_edge = MaskAttentionLoss(criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index), loss_weights=ce_edge_weights)
        self.dice_loss = DiceLoss(apply_softmax=True, ignore_index=None if ignore_index < 0 else ignore_index)

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        names = ["main_loss"]
        # Append aux losses names
        names += [f"aux_loss{i}" for i in range(self.num_aux_heads)]
        # Append detail losses names
        names += [f"detail_loss{i}" for i in range(self.num_detail_heads)]
        names += ["loss"]
        return names

    def forward(self, preds: Tuple[torch.Tensor], target: torch.Tensor):
        """
        :param preds: Model output predictions, must be in the followed format:
         [Main-feats, Aux-feats[0], ..., Aux-feats[num_auxs-1], Detail-feats[0], ..., Detail-feats[num_details-1]
        """
        assert (
            len(preds) == self.num_aux_heads + self.num_detail_heads + 1
        ), f"Wrong num of predictions tensors, expected {self.num_aux_heads + self.num_detail_heads + 1} found {len(preds)}"

        edge_target = target_to_binary_edge(
            target, num_classes=self.num_classes, kernel_size=self.edge_kernel, ignore_index=self.ignore_index, flatten_channels=True
        )
        losses = []
        total_loss = 0
        # Main and auxiliaries feature maps losses
        for i in range(0, 1 + self.num_aux_heads):
            ce_loss = self.ce_edge(preds[i], target, edge_target)
            dice_loss = self.dice_loss(preds[i], target)

            loss = ce_loss * self.dice_ce_weights[0] + dice_loss * self.dice_ce_weights[1]
            total_loss += self.weights[i] * loss
            losses.append(loss)

        # Detail feature maps losses
        if self.use_detail:
            for i in range(1 + self.num_aux_heads, len(preds)):
                bce_loss = self.bce(preds[i], edge_target)
                dice_loss = self.binary_dice(preds[i], edge_target)

                loss = bce_loss * self.dice_ce_weights[0] + dice_loss * self.dice_ce_weights[1]
                total_loss += self.weights[i] * loss
                losses.append(loss)

        losses.append(total_loss)

        return total_loss, torch.stack(losses, dim=0).detach()
