from typing import Optional

import torch
from torch import Tensor

__all__ = ["batch_distance2bbox"]


def batch_distance2bbox(points: Tensor, distance: Tensor, max_shapes: Optional[Tensor] = None) -> Tensor:
    """Decode distance prediction to bounding box for batch.

    :param points: [B, ..., 2], "xy" format
    :param distance: [B, ..., 4], "ltrb" format
    :param max_shapes: [B, 2], "h,w" format, Shape of the image.
    :return: Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, dim=-1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], dim=-1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox
