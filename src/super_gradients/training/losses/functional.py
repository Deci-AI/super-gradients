import math
from typing import Tuple

import torch
from torch import Tensor


def bbox_overlap(box1: Tuple[Tensor, Tensor, Tensor, Tensor], box2: Tuple[Tensor, Tensor, Tensor, Tensor], eps: float = 1e-10) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Calculate the iou of box1 and box2.

    :param box1:    box1 with the shape (..., 4)
    :param box2:    box1 with the shape (..., 4)
    :param eps:     epsilon to avoid divide by zero
    :return:
        - iou:      iou of box1 and box2
        - overlap:  overlap of box1 and box2
        - union:    union of box1 and box2
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xkis1 = torch.maximum(x1, x1g)
    ykis1 = torch.maximum(y1, y1g)
    xkis2 = torch.minimum(x2, x2g)
    ykis2 = torch.minimum(y2, y2g)
    w_inter = (xkis2 - xkis1).clip(0)
    h_inter = (ykis2 - ykis1).clip(0)
    overlap = w_inter * h_inter

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - overlap + eps
    iou = overlap / union

    return iou, overlap, union


def bbox_ciou_loss(pred_bboxes: Tensor, target_bboxes: Tensor, eps: float) -> Tensor:
    """
    Compute CIoU loss between predicted and target bboxes.
    :param pred_bboxes:   Predicted boxes in xyxy format of [D0, D1,...Di, 4] shape
    :param target_bboxes: Target boxes in xyxy format of [D0, D1,...Di, 4] shape
    :return: CIoU loss per each box as tensor of shape [D0, D1,...Di]
    """

    x1, y1, x2, y2 = pred_bboxes.chunk(4, dim=-1)
    x1g, y1g, x2g, y2g = target_bboxes.chunk(4, dim=-1)

    box1 = [x1, y1, x2, y2]
    box2 = [x1g, y1g, x2g, y2g]
    iou, overlap, union = bbox_overlap(box1, box2, eps)
    xc1 = torch.minimum(x1, x1g)
    yc1 = torch.minimum(y1, y1g)
    xc2 = torch.maximum(x2, x2g)
    yc2 = torch.maximum(y2, y2g)

    w1 = xc2 - xc1
    h1 = yc2 - yc1
    w2 = x2g - x1g
    h2 = y2g - y1g

    area_c = (xc2 - xc1) * (yc2 - yc1) + eps

    iou = iou - ((area_c - union) / area_c)

    # convex diagonal squared

    c2 = cw**2 + ch**2 + eps  # noqa

    # centerpoint distance squared
    rho2 = ((x1g + x2g - x1 - x2) ** 2 + (y1g + y2g - y1 - y2) ** 2) / 4

    v = (4 / math.pi**2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / ((1 + eps) - iou + v)
    iou -= rho2 / c2 + v * alpha  # CIoU

    return 1 - iou
