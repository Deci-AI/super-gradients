from typing import Union

import torch
from torch import Tensor
import torch.nn.functional as F


class ShiftAndScaleInvariantLoss(torch.nn.Module):
    """
    Shift & scale invariant robust loss function
    """

    def __init__(self, keep_fraction: float = 0.8, ignore_value: float = float("nan")) -> None:
        super().__init__()
        self.keep_fraction = keep_fraction
        self.ignore_value = ignore_value

    @torch.cuda.amp.autocast(False)
    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        if output.size() != target.size():
            raise ValueError(f"Size of target({target.size()}) and output({output.size()}) must match")

        output_norm = self.normalize_depth_with_ignore(output, ignore_value=self.ignore_value)
        target_norm = self.normalize_depth_with_ignore(target, ignore_value=self.ignore_value)
        output_norm = output_norm.view(output_norm.size(0), -1)
        target_norm = target_norm.view(target_norm.size(0), -1)

        loss = F.l1_loss(output_norm, target_norm, reduction="none")
        _, M = loss.size(0), loss.size(1)

        ignore_mask = ~torch.isfinite(target_norm) | target_norm.eq(self.ignore_value)
        loss = torch.masked_fill(loss, ignore_mask, 0)

        if self.keep_fraction < 1:
            num_elements_to_keep = int(M * self.keep_fraction)
            loss, _ = torch.topk(loss, k=num_elements_to_keep, dim=1, largest=False, sorted=False)

        # Original formula
        # return mae.sum(dtype=torch.float32) / (2 * batch_size * M)
        # A simple mean would be more numerically stable
        return loss.mean()

    @staticmethod
    def normalize_depth_with_ignore(d: Tensor, ignore_value: Union[int, float] = float("nan")) -> Tensor:
        """
        Normalize depth map to to have zero translation and unit scale.
        Equation 5,6 from https://arxiv.org/pdf/1907.01341v3.pdf

        Args:
            d: Depth, inverse depth or disparity map (B,1,H,W)

        Returns:

        """
        if d.size(1) != 1:
            raise ValueError("Number of channels in depth map must be 1")

        ignore_mask = (~torch.isfinite(d)) | d.eq(ignore_value)

        bs = d.size(0)
        depths = []

        for i in range(bs):
            not_ignored_mask = ~ignore_mask[i]
            depth = d[i]
            not_ignored_depth = depth[not_ignored_mask]
            if len(not_ignored_depth):
                median = torch.median(not_ignored_depth)
                scale = (depth - median)[not_ignored_mask]
                scale = scale.abs().mean()

                depth = (depth - median) / (scale + 1e-5)
                depth[~not_ignored_mask] = ignore_value

            depths.append(depth)

        depths = torch.stack(depths)
        return depths
