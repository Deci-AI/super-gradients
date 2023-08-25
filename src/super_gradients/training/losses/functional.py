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

    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, dim=-1)

    box1 = [b1_x1, b1_y1, b1_x2, b1_y2]
    box2 = [b2_x1, b2_y1, b2_x2, b2_y2]
    iou, overlap, union = bbox_overlap(box1, box2, eps)
    xc1 = torch.minimum(b1_x1, b2_x1)
    yc1 = torch.minimum(b1_y1, b2_y1)
    xc2 = torch.maximum(b1_x2, b2_x2)
    yc2 = torch.maximum(b1_y2, b2_y2)

    w1 = xc2 - xc1
    h1 = yc2 - yc1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1

    area_c = (xc2 - xc1) * (yc2 - yc1) + eps

    iou = iou - ((area_c - union) / area_c)

    # convex diagonal squared

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

    c2 = cw**2 + ch**2 + eps  # noqa

    # centerpoint distance squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    v = (4 / math.pi**2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / ((1 + eps) - iou + v)
    iou -= rho2 / c2 + v * alpha  # CIoU

    return 1 - iou
