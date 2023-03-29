from abc import ABC, abstractmethod
from typing import Union, Optional

import torch
from torch.nn.modules.loss import _Loss

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.losses.loss_utils import apply_reduce, LossReduction
from super_gradients.training.utils.segmentation_utils import to_one_hot

logger = get_logger(__name__)


class AbstarctSegmentationStructureLoss(_Loss, ABC):
    """
    Abstract computation of structure loss between two tensors, It can support both multi-classes and binary tasks.
    """

    def __init__(
        self,
        apply_softmax: bool = True,
        ignore_index: int = None,
        smooth: float = 1.0,
        eps: float = 1e-5,
        reduce_over_batches: bool = False,
        generalized_metric: bool = False,
        weight: Optional[torch.Tensor] = None,
        reduction: Union[LossReduction, str] = "mean",
    ):
        """
        :param apply_softmax: Whether to apply softmax to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the metric
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        :param reduce_over_batches: Whether to average metric over the batch axis if set True,
         default is `False` to average over the classes axis.
        :param generalized_metric: Whether to apply normalization by the volume of each class.
        :param weight: a manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`.
        :param reduction: Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
            `none`: no reduction will be applied.
            `mean`: the sum of the output will be divided by the number of elements in the output.
            `sum`: the output will be summed.
            Default: `mean`
        """
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.apply_softmax = apply_softmax
        self.eps = eps
        self.smooth = smooth
        self.reduce_over_batches = reduce_over_batches
        self.generalized_metric = generalized_metric
        self.weight = weight
        if self.generalized_metric:
            assert self.weight is None, "Cannot use structured Loss with weight classes and generalized normalization"
            if self.eps > 1e-12:
                logger.warning("When using GeneralizedLoss, it is recommended to use eps below 1e-12, to not affect" "small values normalized terms.")
            if self.smooth != 0:
                logger.warning("When using GeneralizedLoss, it is recommended to set smooth value as 0.")

    @abstractmethod
    def _calc_numerator_denominator(self, labels_one_hot, predict) -> (torch.Tensor, torch.Tensor):
        """
        All base classes must implement this function.
        Return: 2 tensor of shape [BS, num_classes, img_width, img_height].
        """
        raise NotImplementedError()

    @abstractmethod
    def _calc_loss(self, numerator, denominator) -> torch.Tensor:
        """
        All base classes must implement this function.
        Return a tensors of shape [BS] if self.reduce_over_batches else [num_classes].
        """
        raise NotImplementedError()

    def forward(self, predict, target):
        if self.apply_softmax:
            predict = torch.softmax(predict, dim=1)
        # target to one hot format
        if target.size() == predict.size():
            labels_one_hot = target
        elif target.dim() == 3:  # if target tensor is in class indexes format.
            if predict.size(1) == 1 and self.ignore_index is None:  # if one class prediction task
                labels_one_hot = target.unsqueeze(1)
            else:
                labels_one_hot = to_one_hot(target, num_classes=predict.shape[1], ignore_index=self.ignore_index)
        else:
            raise AssertionError(
                f"Mismatch of target shape: {target.size()} and prediction shape: {predict.size()},"
                f" target must be [NxWxH] tensor for to_one_hot conversion"
                f" or to have the same num of channels like prediction tensor"
            )

        reduce_spatial_dims = list(range(2, len(predict.shape)))
        reduce_dims = [1] + reduce_spatial_dims if self.reduce_over_batches else [0] + reduce_spatial_dims

        # Calculate the numerator and denominator of the chosen metric
        numerator, denominator = self._calc_numerator_denominator(labels_one_hot, predict)

        # exclude ignore labels from numerator and denominator, false positive predicted on ignore samples
        # are not included in the total calculation.
        if self.ignore_index is not None:
            valid_mask = target.ne(self.ignore_index).unsqueeze(1).expand_as(denominator)
            numerator *= valid_mask
            denominator *= valid_mask

        numerator = torch.sum(numerator, dim=reduce_dims)
        denominator = torch.sum(denominator, dim=reduce_dims)

        if self.generalized_metric:
            weights = 1.0 / (torch.sum(labels_one_hot, dim=reduce_dims) ** 2)
            # if some classes are not in batch, weights will be inf.
            infs = torch.isinf(weights)
            weights[infs] = 0.0
            numerator *= weights
            denominator *= weights

        # Calculate the loss of the chosen metric
        losses = self._calc_loss(numerator, denominator)
        if self.weight is not None:
            losses *= self.weight
        return apply_reduce(losses, reduction=self.reduction)
