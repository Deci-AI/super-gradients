from typing import Union

import torch

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.losses.loss_utils import LossReduction
from super_gradients.training.losses.structure_loss import AbstarctSegmentationStructureLoss

logger = get_logger(__name__)


class DiceLoss(AbstarctSegmentationStructureLoss):
    """
    Compute average Dice loss between two tensors, It can support both multi-classes and binary tasks.
    Defined in the paper: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
    """
    def _calc_numerator_denominator(self, labels_one_hot, predict):
        """
        Calculate dice metric's numerator and denominator.

        :param labels_one_hot: target in one hot format.   shape: [BS, num_classes, img_width, img_height]
        :param predict: predictions tensor.                shape: [BS, num_classes, img_width, img_height]
        :return:
            numerator = intersection between predictions and target. shape: [BS, num_classes, img_width, img_height]
            denominator = sum of predictions and target areas.       shape: [BS, num_classes, img_width, img_height]
        """
        numerator = labels_one_hot * predict
        denominator = labels_one_hot + predict
        return numerator, denominator

    def _calc_loss(self, numerator, denominator):
        """
        Calculate dice loss.
        All tensors are of shape [BS] if self.reduce_over_batches else [num_classes].

        :param numerator: intersection between predictions and target.
        :param denominator: total number of pixels of prediction and target.
        """
        loss = 1. - ((2. * numerator + self.smooth) / (denominator + self.eps + self.smooth))
        return loss


class BinaryDiceLoss(DiceLoss):
    """
    Compute Dice Loss for binary class tasks (1 class only).
    Except target to be a binary map with 0 and 1 values.
    """
    def __init__(self,
                 apply_sigmoid: bool = True,
                 smooth: float = 1.,
                 eps: float = 1e-5):
        """
        :param apply_sigmoid: Whether to apply sigmoid to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the dice
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        """
        super().__init__(apply_softmax=False, ignore_index=None, smooth=smooth, eps=eps, reduce_over_batches=False)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, predict, target):
        if self.apply_sigmoid:
            predict = torch.sigmoid(predict)
        return super().forward(predict=predict, target=target)


class GeneralizedDiceLoss(DiceLoss):
    """
    Compute the Generalised Dice loss, contribution of each label is normalized by the inverse of its volume, in order
     to deal with class imbalance.
    Defined in the paper: "Generalised Dice overlap as a deep learning loss function for highly unbalanced
     segmentations"
    Args:
        smooth (float): default value is 0, smooth laplacian is not recommended to be used with GeneralizedDiceLoss.
         because the weighted values to be added are very small.
        eps (float): default value is 1e-17, must be a very small value, because weighted `intersection` and
        `denominator` are very small after multiplication with `1 / counts ** 2`
    """
    def __init__(self,
                 apply_softmax: bool = True,
                 ignore_index: int = None,
                 smooth: float = 0.0,
                 eps: float = 1e-17,
                 reduce_over_batches: bool = False,
                 reduction: Union[LossReduction, str] = "mean"
                 ):
        """
        :param apply_softmax: Whether to apply softmax to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the dice
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        :param reduce_over_batches: Whether to apply reduction over the batch axis if set True,
         default is `False` to average over the classes axis.
        :param reduction: Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
            `none`: no reduction will be applied.
            `mean`: the sum of the output will be divided by the number of elements in the output.
            `sum`: the output will be summed.
            Default: `mean`
        """
        super().__init__(apply_softmax=apply_softmax, ignore_index=ignore_index, smooth=smooth, eps=eps,
                         reduce_over_batches=reduce_over_batches, generalized_metric=True, weight=None,
                         reduction=reduction)
