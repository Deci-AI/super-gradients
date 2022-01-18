import torch
from typing import Union, Optional
from torch.nn.modules.loss import _Loss
from super_gradients.training.utils.segmentation_utils import to_one_hot
from super_gradients.training.losses.loss_utils import apply_reduce, LossReduction
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors, It can support both multi-classes and binary tasks.
    Defined in the paper: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
    """
    def __init__(self,
                 apply_softmax: bool = True,
                 ignore_index: int = None,
                 smooth: float = 1.,
                 eps: float = 1e-5,
                 sum_over_batches: bool = False,
                 generalized_dice: bool = False,
                 weight: Optional[torch.Tensor] = None,
                 reduction: Union[LossReduction, str] = "mean"):
        """
        :param apply_softmax: Whether to apply softmax to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the dice
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        :param sum_over_batches: Whether to average dice over the batch axis if set True,
         default is `False` to average over the classes axis.
        :param generalized_dice: Whether to apply normalization by the volume of each class.
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
        self.sum_over_batches = sum_over_batches
        self.generalized_dice = generalized_dice
        self.weight = weight
        if self.generalized_dice:
            assert self.weight is None, "Cannot use Dice Loss with weight classes and generalized dice normalization"
            if self.eps > 1e-12:
                logger.warning("When using GeneralizedDiceLoss, it is recommended to use eps below 1e-12, to not affect"
                               "small values normalized terms.")
            if self.smooth != 0:
                logger.warning("When using GeneralizedDiceLoss, it is recommended to set smooth value as 0.")

    def forward(self, predict, target):
        if self.apply_softmax:
            predict = torch.softmax(predict, dim=1)
        # target to one hot format
        if target.size() == predict.size():
            labels_one_hot = target
        elif len(target.size()) == 3:       # if target tensor is in class indexes format.
            if predict.size(1) == 1 and self.ignore_index is None:    # if one class prediction task
                labels_one_hot = target.unsqueeze(1)
            else:
                labels_one_hot = to_one_hot(target, num_classes=predict.shape[1], ignore_index=self.ignore_index)
        else:
            raise AssertionError(f"Mismatch of target shape: {target.size()} and prediction shape: {predict.size()},"
                                 f" target must be [NxWxH] tensor for to_one_hot conversion"
                                 f" or to have the same num of channels like prediction tensor")

        reduce_spatial_dims = list(range(2, len(predict.shape)))
        reduce_dims = [1] + reduce_spatial_dims if self.sum_over_batches else [0] + reduce_spatial_dims

        intersection = torch.sum(labels_one_hot * predict, dim=reduce_dims)
        denominator = labels_one_hot + predict
        # exclude ignore labels from denominator, false positive predicted on ignore samples are not included in
        # total denominator.
        if self.ignore_index is not None:
            valid_mask = target.ne(self.ignore_index).unsqueeze(1).expand_as(denominator)
            denominator *= valid_mask
        denominator = torch.sum(denominator, dim=reduce_dims)

        if self.generalized_dice:
            weights = 1. / (torch.sum(labels_one_hot, dim=reduce_dims) ** 2)
            # if some classes are not in batch, weights will be inf.
            infs = torch.isinf(weights)
            weights[infs] = 0.0
            intersection *= weights
            denominator *= weights

        dices = 1. - ((2. * intersection + self.smooth) / (denominator + self.eps + self.smooth))
        if self.weight is not None:
            dices *= self.weight
        return apply_reduce(dices, reduction=self.reduction)


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
        super().__init__(apply_softmax=False, ignore_index=None, smooth=smooth, eps=eps, sum_over_batches=False)
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
                 sum_over_batches: bool = False,
                 reduction: Union[LossReduction, str] = "mean"
                 ):
        """
        :param apply_softmax: Whether to apply softmax to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the dice
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        :param sum_over_batches: Whether to average dice over the batch axis if set True,
         default is `False` to average over the classes axis.
        :param reduction: Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
            `none`: no reduction will be applied.
            `mean`: the sum of the output will be divided by the number of elements in the output.
            `sum`: the output will be summed.
            Default: `mean`
        """
        super().__init__(apply_softmax=apply_softmax, ignore_index=ignore_index, smooth=smooth, eps=eps,
                         sum_over_batches=sum_over_batches, generalized_dice=True, weight=None, reduction=reduction)
