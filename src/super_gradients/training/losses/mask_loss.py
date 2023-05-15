import torch
from torch.nn.modules.loss import _Loss
from super_gradients.training.losses.loss_utils import apply_reduce, LossReduction
from typing import Union


class MaskAttentionLoss(_Loss):
    """
    Pixel mask attention loss. For semantic segmentation usages with 4D tensors.
    """

    def __init__(self, criterion: _Loss, loss_weights: Union[list, tuple] = (1.0, 1.0), reduction: Union[LossReduction, str] = "mean"):
        """
        :param criterion: _Loss object, loss function that apply per pixel cost penalty are supported, i.e
            CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, SL1Loss.
            criterion reduction must be `none`.
        :param loss_weights: Weight to apply for each part of the loss contributions,
            [regular loss, masked loss] respectively.
        :param reduction: Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
            `none`: no reduction will be applied.
            `mean`: the sum of the output will be divided by the number of elements in the output.
            `sum`: the output will be summed.
            Default: `mean`
        """
        super().__init__(reduction=reduction.value if isinstance(reduction, LossReduction) else reduction)
        # Check that the arguments are valid.
        if criterion.reduction != "none":
            raise ValueError(f"criterion reduction must be `none`, for computing the mask contribution loss values," f" found reduction: {criterion.reduction}")
        if len(loss_weights) != 2:
            raise ValueError(f"loss_weights must have 2 values, found: {len(loss_weights)}")
        if loss_weights[1] <= 0:
            raise ValueError("If no loss weight is applied on mask samples, consider using simply criterion")

        self.criterion = criterion
        self.loss_weights = loss_weights

    def forward(self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        criterion_loss = self.criterion(predict, target)

        mask = self._broadcast_mask(mask, criterion_loss.size())
        mask_loss = criterion_loss * mask

        if self.reduction == LossReduction.NONE.value:
            return criterion_loss * self.loss_weights[0] + mask_loss * self.loss_weights[1]
        mask_loss = mask_loss[mask == 1]  # consider only mask samples for mask loss computing
        # If mask doesn't include foreground values, set mask_loss as 0.
        if mask_loss.numel() == 0:
            mask_loss = torch.tensor(0.0)

        mask_loss = apply_reduce(mask_loss, self.reduction)
        criterion_loss = apply_reduce(criterion_loss, self.reduction)

        loss = criterion_loss * self.loss_weights[0] + mask_loss * self.loss_weights[1]
        return loss

    def _broadcast_mask(self, mask: torch.Tensor, size: torch.Size):
        """
        Broadcast the mask tensor before elementwise multiplication.
        """
        # Assert that batch size and spatial size are the same.
        if mask.size()[-2:] != size[-2:] or mask.size(0) != size[0]:
            raise AssertionError(
                "Mask broadcast is allowed only in channels dimension, found shape mismatch between" f"mask shape: {mask.size()}, and target shape: {size}"
            )
        # when mask is [B, 1, H, W] | [B, H, W] and size is [B, H, W]
        # or when mask is [B, 1, H, W] | [B, H, W] and size is [B, 1, H, W]
        if len(size) == 3 or (len(size) == 4 and size[1] == 1):
            mask = mask.view(*size)

        # when mask is [B, C, H, W] | [B, 1, H, W] | [B, H, W] and size is [B, C, H, W]
        else:
            mask = mask if len(mask.size()) == 4 else mask.unsqueeze(1)
            if mask.size(1) not in [1, size[1]]:
                raise AssertionError(
                    f"Broadcast is not allowed, num mask channels must be 1 or same as target channels" f"mask shape: {mask.size()}, and target shape: {size}"
                )
            mask = mask if mask.size() == size else mask.expand(*size)
        return mask
