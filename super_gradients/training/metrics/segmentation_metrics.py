import numpy as np
import torch
import torchmetrics
from torchmetrics import Metric


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


class PixelAccuracy(Metric):
    def __init__(self, ignore_label=-100, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_label = ignore_label
        self.add_state("total_correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_label", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if isinstance(preds, tuple):
            preds = preds[0]
        _, predict = torch.max(preds, 1)
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


class IoU(torchmetrics.IoU):
    def __init__(self, num_classes, dist_sync_on_step=True, ignore_index=None):
        super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step, ignore_index=ignore_index)

    def update(self, preds, target: torch.Tensor):
        # WHEN DEALING WITH MULTIPLE OUTPUTS- OUTPUTS[0] IS THE MAIN SEGMENTATION MAP
        if isinstance(preds, tuple):
            preds = preds[0]
        _, preds = torch.max(preds, 1)
        super().update(preds=preds, target=target)
