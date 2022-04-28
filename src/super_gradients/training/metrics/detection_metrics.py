import numpy as np
import torch
from torchmetrics import Metric
import super_gradients


def compute_ap(recall, precision, method: str = 'interp'):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        :param recall:    The recall curve - ndarray [1, points in curve]
        :param precision: The precision curve - ndarray [1, points in curve]
        :param method: 'continuous', 'interp'
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # IN ORDER TO CALCULATE, WE HAVE TO MAKE SURE THE CURVES GO ALL THE WAY TO THE AXES (FROM X=0 TO Y=0)
    # THIS IS HOW IT IS COMPUTED IN  ORIGINAL REPO - A MORE CORRECT COMPUTE WOULD BE ([0.], recall, [recall[-1] + 1E-3])
    wrapped_recall = np.concatenate(([0.], recall, [1.0]))
    wrapped_precision = np.concatenate(([1.], precision, [0.]))

    # COMPUTE THE PRECISION ENVELOPE
    wrapped_precision = np.flip(np.maximum.accumulate(np.flip(wrapped_precision)))

    # INTEGRATE AREA UNDER CURVE
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, wrapped_recall, wrapped_precision), x)  # integrate
    else:  # 'continuous'
        i = np.where(wrapped_recall[1:] != wrapped_recall[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((wrapped_recall[i + 1] - wrapped_recall[i]) * wrapped_precision[i + 1])  # area under curve

    return ap

