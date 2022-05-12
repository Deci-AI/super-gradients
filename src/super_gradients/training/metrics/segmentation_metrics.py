import numpy as np
import torch
import torchmetrics
from torchmetrics import Metric
from typing import Optional, Tuple
from torchmetrics.utilities.distributed import reduce
from abc import ABC, abstractmethod


def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(predict, 1)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(predict, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def _dice_from_confmat(
    confmat: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    reduction: str = "elementwise_mean",
) -> torch.Tensor:
    """Computes Dice coefficient from confusion matrix.

    Args:
        confmat: Confusion matrix without normalization
        num_classes: Number of classes for a given prediction and target tensor
        ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method.
        absent_score: score to use for an individual class, if no instances of the class index were present in `pred`
            AND no instances of the class index were present in `target`.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied
    """

    # Remove the ignored class index from the scores.
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        confmat[ignore_index] = 0.0

    intersection = torch.diag(confmat)
    denominator = confmat.sum(0) + confmat.sum(1)

    # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
    scores = 2 * intersection.float() / denominator.float()
    scores[denominator == 0] = absent_score

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        scores = torch.cat(
            [
                scores[:ignore_index],
                scores[ignore_index + 1:],
            ]
        )

    return reduce(scores, reduction=reduction)


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


class AbstractMetricsArgsPrepFn(ABC):
    """
    Abstract preprocess metrics arguments class.
    """
    @abstractmethod
    def __call__(self, preds, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        All base classes must implement this function and return a tuple of torch tensors (predictions, target).
        """
        raise NotImplementedError()


class PreprocessSegmentationMetricsArgs(AbstractMetricsArgsPrepFn):
    """
    Default segmentation inputs preprocess function before updating segmentation metrics, handles multiple inputs and
    apply normalizations.
    """
    def __init__(self,
                 apply_arg_max: bool = False,
                 apply_sigmoid: bool = False):
        """
        :param apply_arg_max: Whether to apply argmax on predictions tensor.
        :param apply_sigmoid:  Whether to apply sigmoid on predictions tensor.
        """
        self.apply_arg_max = apply_arg_max
        self.apply_sigmoid = apply_sigmoid

    def __call__(self, preds, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # WHEN DEALING WITH MULTIPLE OUTPUTS- OUTPUTS[0] IS THE MAIN SEGMENTATION MAP
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if self.apply_arg_max:
            _, preds = torch.max(preds, 1)
        elif self.apply_sigmoid:
            preds = torch.sigmoid(preds)

        target = target.long()
        return preds, target


class PixelAccuracy(Metric):
    def __init__(self,
                 ignore_label=-100,
                 dist_sync_on_step=False,
                 metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_label = ignore_label
        self.add_state("total_correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_label", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_arg_max=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        predict, target = self.metrics_args_prep_fn(preds, target)

        labeled_mask = target.ne(self.ignore_label)
        pixel_labeled = torch.sum(labeled_mask)
        pixel_correct = torch.sum((predict == target) * labeled_mask)
        self.total_correct += pixel_correct
        self.total_label += pixel_labeled

    def compute(self):
        _total_correct = self.total_correct.cpu().detach().numpy().astype('int64')
        _total_label = self.total_label.cpu().detach().numpy().astype('int64')
        pix_acc = np.float64(1.0) * _total_correct / (np.spacing(1, dtype=np.float64) + _total_label)
        return pix_acc


class IoU(torchmetrics.JaccardIndex):
    def __init__(self,
                 num_classes: int,
                 dist_sync_on_step: bool = False,
                 ignore_index: Optional[int] = None,
                 reduction: str = "elementwise_mean",
                 threshold: float = 0.5,
                 metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None):
        super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index,
                         reduction=reduction, threshold=threshold)
        self.metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_arg_max=True)

    def update(self, preds, target: torch.Tensor):
        preds, target = self.metrics_args_prep_fn(preds, target)
        super().update(preds=preds, target=target)


class Dice(torchmetrics.JaccardIndex):
    def __init__(self,
                 num_classes: int,
                 dist_sync_on_step: bool = False,
                 ignore_index: Optional[int] = None,
                 reduction: str = "elementwise_mean",
                 threshold: float = 0.5,
                 metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None):
        super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index,
                         reduction=reduction, threshold=threshold)
        self.metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_arg_max=True)

    def update(self, preds, target: torch.Tensor):
        preds, target = self.metrics_args_prep_fn(preds, target)
        super().update(preds=preds, target=target)

    def compute(self) -> torch.Tensor:
        """Computes Dice coefficient"""
        return _dice_from_confmat(
            self.confmat, self.num_classes, self.ignore_index, self.absent_score, self.reduction
        )


class BinaryIOU(IoU):
    def __init__(self,
                 dist_sync_on_step=True,
                 ignore_index: Optional[int] = None,
                 threshold: float = 0.5,
                 metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None):
        metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_sigmoid=True)
        super().__init__(num_classes=2, dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index,
                         reduction="none", threshold=threshold, metrics_args_prep_fn=metrics_args_prep_fn)
        self.component_names = ["target_IOU", "background_IOU", "mean_IOU"]

    def compute(self):
        ious = super(BinaryIOU, self).compute()
        return {"target_IOU": ious[1], "background_IOU": ious[0], "mean_IOU": ious.mean()}


class BinaryDice(Dice):
    def __init__(self,
                 dist_sync_on_step=True,
                 ignore_index: Optional[int] = None,
                 threshold: float = 0.5,
                 metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None):
        metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_sigmoid=True)
        super().__init__(num_classes=2, dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index,
                         reduction="none", threshold=threshold, metrics_args_prep_fn=metrics_args_prep_fn)
        self.component_names = ["target_Dice", "background_Dice", "mean_Dice"]

    def compute(self):
        dices = super().compute()
        return {"target_Dice": dices[1], "background_Dice": dices[0], "mean_Dice": dices.mean()}
