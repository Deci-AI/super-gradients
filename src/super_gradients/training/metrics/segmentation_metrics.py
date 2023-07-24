import typing

import numpy as np
import torch
import torchmetrics
from torchmetrics import Metric
from typing import Optional, Tuple, List, Union
from torchmetrics.utilities.distributed import reduce
from abc import ABC, abstractmethod


from super_gradients.common.object_names import Metrics
from super_gradients.common.registry.registry import register_metric


def batch_pix_accuracy(predict: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Batch Pixel Accuracy

    :param predict: input 4D tensor
    :param target: label 3D tensor
    """
    _, predict = torch.max(predict, 1)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict: torch.Tensor, target: torch.Tensor, nclass: int) -> Tuple[float, float]:
    """Batch Intersection of Union

    :param predict: input 4D tensor
    :param target: label 3D tensor
    :param nclass: number of categories (int)
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
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
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

    :param confmat:         Confusion matrix without normalization
    :param num_classes:     Number of classes for a given prediction and target tensor
    :param ignore_index:    Optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method.
    :param absent_score:    Score to use for an individual class, if no instances of the class index were present in `pred`
            AND no instances of the class index were present in `target`.
    :param reduction:       Method to reduce metric score over labels.
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
                scores[ignore_index + 1 :],
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
    area_inter, _ = np.histogram(intersection, bins=num_class - 1, range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1, range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1, range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


def _map_ignored_inds(target: torch.Tensor, ignore_index_list: List[int], unfiltered_num_classes: int) -> torch.Tensor:
    """
    Creaetes a copy of target, mapping indices in range(unfiltered_num_classes) to range(unfiltered_num_classes-len(
    ignore_index_list)+1). Indices in ignore_index_list are being mapped to 0, which can later on be used as
     "ignore_index".

     Example:
        >>>_map_ignored_inds(torch.tensor([0,1,2,3,4,5,6]), ignore_index_list=[3,5,1], unfiltered_num_classes=7)
        >>> tensor([1, 0, 2, 0, 3, 0, 4])



    :param target: torch.Tensor, tensor to perform the mapping on.
    :param ignore_index_list: List[int], list of indices to map to 0 in the output tensor.
    :param unfiltered_num_classes: int, Total number of possible class indices in target.

    :return: mapped tensor as described above.
    """
    target_copy = torch.zeros_like(target)
    all_unfiltered_classes = list(range(unfiltered_num_classes))
    filtered_classes = [i for i in all_unfiltered_classes if i not in ignore_index_list]
    for mapped_idx in range(len(filtered_classes)):
        cls_to_map = filtered_classes[mapped_idx]
        map_val = mapped_idx + 1
        target_copy[target == cls_to_map] = map_val

    return target_copy


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

    def __init__(self, apply_arg_max: bool = False, apply_sigmoid: bool = False):
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


@register_metric(Metrics.PIXEL_ACCURACY)
class PixelAccuracy(Metric):
    """
    Pixel Accuracy

    Args:
        ignore_label: Optional[Union[int, List[int]]], specifying a target class(es) to ignore.
            If given, this class index does not contribute to the returned score, regardless of reduction method.
            Has no effect if given an int that is not in the range [0, num_classes-1].
            By default, no index is ignored, and all classes are used.
            IMPORTANT: reduction="none" alongside with a list of ignored indices is not supported and will raise an error.
        reduction: a method to reduce metric score over labels:

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        metrics_args_prep_fn: Callable, inputs preprocess function applied on preds, target before updating metrics.
            By default set to PreprocessSegmentationMetricsArgs(apply_arg_max=True)
    """

    def __init__(self, ignore_label: Union[int, List[int]] = -100, dist_sync_on_step=False, metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_label = ignore_label
        self.greater_is_better = True
        self.add_state("total_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_label", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_arg_max=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        predict, target = self.metrics_args_prep_fn(preds, target)
        labeled_mask = self._handle_multiple_ignored_inds(target)

        pixel_labeled = torch.sum(labeled_mask)
        pixel_correct = torch.sum((predict == target) * labeled_mask)
        self.total_correct += pixel_correct
        self.total_label += pixel_labeled

    def _handle_multiple_ignored_inds(self, target):
        if isinstance(self.ignore_label, typing.Iterable):
            evaluated_classes_mask = torch.ones_like(target)
            for ignored_label in self.ignore_label:
                evaluated_classes_mask = evaluated_classes_mask.masked_fill(target.eq(ignored_label), 0)
        else:
            evaluated_classes_mask = target.ne(self.ignore_label)

        return evaluated_classes_mask

    def compute(self):
        _total_correct = self.total_correct.cpu().detach().numpy().astype("int64")
        _total_label = self.total_label.cpu().detach().numpy().astype("int64")
        pix_acc = np.float64(1.0) * _total_correct / (np.spacing(1, dtype=np.float64) + _total_label)
        return pix_acc


def _handle_multiple_ignored_inds(ignore_index: Union[int, List[int]], num_classes: int):
    """
    Helper method for variable assignment, prior to the

    super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index, reduction=reduction, threshold=threshold)

    call in segmentation metrics inheriting from torchmetrics.JaccardIndex.
    When ignore_index is list, the num_classes being passed to the torchmetrics.JaccardIndex c'tor is set to be the one after
     mapping of the ignored indices in ignore_index_list to 0. Hence, we set:
      ignore_index=0,
    And since we map all of the ignored indices to 0, it is if we removed them and introduces a new index:
      num_classes = num_classes - len(ignore_index_list) +1
    Unfiltered num_classes is used in .update() for mapping of the original indice values.
    Sets ignore_index to 0
    :param ignore_index: list or single int representing the class ind(ices) to ignore.
    :param num_classes: int, num_classes (original, before mapping) being passed to segmentation metric classesץ
    :return:ignore_index, ignore_index_list, num_classes, unfiltered_num_classesignore_index, ignore_index_list, num_classes, unfiltered_num_classes
    """
    if isinstance(ignore_index, typing.Iterable):
        ignore_index_list = ignore_index
        unfiltered_num_classes = num_classes
        num_classes = num_classes - len(ignore_index_list) + 1
        ignore_index = 0
    else:
        unfiltered_num_classes = num_classes
        ignore_index_list = None
    return ignore_index, ignore_index_list, num_classes, unfiltered_num_classes


@register_metric(Metrics.IOU)
class IoU(torchmetrics.JaccardIndex):
    """
    IoU Metric

    Args:
        num_classes: Number of classes in the dataset.
        ignore_index: Optional[Union[int, List[int]]], specifying a target class(es) to ignore.
            If given, this class index does not contribute to the returned score, regardless of reduction method.
            Has no effect if given an int that is not in the range [0, num_classes-1].
            By default, no index is ignored, and all classes are used.
            IMPORTANT: reduction="none" alongside with a list of ignored indices is not supported and will raise an error.
        threshold: Threshold value for binary or multi-label probabilities.
        reduction: a method to reduce metric score over labels:

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        metrics_args_prep_fn: Callable, inputs preprocess function applied on preds, target before updating metrics.
            By default set to PreprocessSegmentationMetricsArgs(apply_arg_max=True)
    """

    def __init__(
        self,
        num_classes: int,
        dist_sync_on_step: bool = False,
        ignore_index: Optional[Union[int, List[int]]] = None,
        reduction: str = "elementwise_mean",
        threshold: float = 0.5,
        metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None,
    ):

        if num_classes <= 1:
            raise ValueError(f"IoU class only for multi-class usage! For binary usage, please call {BinaryIOU.__name__}")
        if isinstance(ignore_index, typing.Iterable) and reduction == "none":
            raise ValueError("passing multiple ignore indices ")
        ignore_index, ignore_index_list, num_classes, unfiltered_num_classes = _handle_multiple_ignored_inds(ignore_index, num_classes)

        super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index, reduction=reduction, threshold=threshold)

        self.unfiltered_num_classes = unfiltered_num_classes
        self.ignore_index_list = ignore_index_list
        self.metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_arg_max=True)
        self.greater_is_better = True

    def update(self, preds, target: torch.Tensor):
        preds, target = self.metrics_args_prep_fn(preds, target)
        if self.ignore_index_list is not None:
            target = _map_ignored_inds(target, self.ignore_index_list, self.unfiltered_num_classes)
            preds = _map_ignored_inds(preds, self.ignore_index_list, self.unfiltered_num_classes)
        super().update(preds=preds, target=target)


@register_metric(Metrics.DICE)
class Dice(torchmetrics.JaccardIndex):
    """
    Dice Coefficient Metric

    Args:
        num_classes: Number of classes in the dataset.
        ignore_index: Optional[Union[int, List[int]]], specifying a target class(es) to ignore.
            If given, this class index does not contribute to the returned score, regardless of reduction method.
            Has no effect if given an int that is not in the range [0, num_classes-1].
            By default, no index is ignored, and all classes are used.
            IMPORTANT: reduction="none" alongside with a list of ignored indices is not supported and will raise an error.
        threshold: Threshold value for binary or multi-label probabilities.
        reduction: a method to reduce metric score over labels:

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        metrics_args_prep_fn: Callable, inputs preprocess function applied on preds, target before updating metrics.
            By default set to PreprocessSegmentationMetricsArgs(apply_arg_max=True)
    """

    def __init__(
        self,
        num_classes: int,
        dist_sync_on_step: bool = False,
        ignore_index: Optional[int] = None,
        reduction: str = "elementwise_mean",
        threshold: float = 0.5,
        metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None,
    ):

        if num_classes <= 1:
            raise ValueError(f"Dice class only for multi-class usage! For binary usage, please call {BinaryDice.__name__}")

        ignore_index, ignore_index_list, num_classes, unfiltered_num_classes = _handle_multiple_ignored_inds(ignore_index, num_classes)

        super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index, reduction=reduction, threshold=threshold)

        self.ignore_index_list = ignore_index_list
        self.unfiltered_num_classes = unfiltered_num_classes
        self.metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_arg_max=True)
        self.greater_is_better = True

    def update(self, preds, target: torch.Tensor):
        preds, target = self.metrics_args_prep_fn(preds, target)
        if self.ignore_index_list is not None:
            target = _map_ignored_inds(target, self.ignore_index_list, self.unfiltered_num_classes)
            preds = _map_ignored_inds(preds, self.ignore_index_list, self.unfiltered_num_classes)
        super().update(preds=preds, target=target)

    def compute(self) -> torch.Tensor:
        """Computes Dice coefficient"""
        return _dice_from_confmat(self.confmat, self.num_classes, self.ignore_index, self.absent_score, self.reduction)


@register_metric(Metrics.BINARY_IOU)
class BinaryIOU(IoU):
    def __init__(
        self,
        dist_sync_on_step=True,
        ignore_index: Optional[int] = None,
        threshold: float = 0.5,
        metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None,
    ):
        metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_sigmoid=True)
        super().__init__(
            num_classes=2,
            dist_sync_on_step=dist_sync_on_step,
            ignore_index=ignore_index,
            reduction="none",
            threshold=threshold,
            metrics_args_prep_fn=metrics_args_prep_fn,
        )
        self.greater_component_is_better = {
            "target_IOU": True,
            "background_IOU": True,
            "mean_IOU": True,
        }
        self.component_names = list(self.greater_component_is_better.keys())

    def compute(self):
        ious = super(BinaryIOU, self).compute()
        return {"target_IOU": ious[1], "background_IOU": ious[0], "mean_IOU": ious.mean()}


@register_metric(Metrics.BINARY_DICE)
class BinaryDice(Dice):
    def __init__(
        self,
        dist_sync_on_step=True,
        ignore_index: Optional[int] = None,
        threshold: float = 0.5,
        metrics_args_prep_fn: Optional[AbstractMetricsArgsPrepFn] = None,
    ):
        metrics_args_prep_fn = metrics_args_prep_fn or PreprocessSegmentationMetricsArgs(apply_sigmoid=True)
        super().__init__(
            num_classes=2,
            dist_sync_on_step=dist_sync_on_step,
            ignore_index=ignore_index,
            reduction="none",
            threshold=threshold,
            metrics_args_prep_fn=metrics_args_prep_fn,
        )
        self.greater_component_is_better = {
            "target_Dice": True,
            "background_Dice": True,
            "mean_Dice": True,
        }
        self.component_names = list(self.greater_component_is_better.keys())

    def compute(self):
        dices = super().compute()
        return {"target_Dice": dices[1], "background_Dice": dices[0], "mean_Dice": dices.mean()}
