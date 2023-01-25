from typing import Tuple, Dict

import torch
from torch.utils.data._utils.collate import default_collate


class DetectionCollateFN:
    """
    Collate function for Yolox training
    """

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = default_collate(data)
        ims, targets = batch[0:2]
        return ims, self._format_targets(targets)

    def _format_targets(self, targets: torch.Tensor) -> torch.Tensor:
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)  # number of label per image
        targets_merged = []
        for i in range(targets.shape[0]):
            targets_im = targets[i, : nlabel[i]]
            batch_column = targets.new_ones((targets_im.shape[0], 1)) * i
            targets_merged.append(torch.cat((batch_column, targets_im), 1))
        return torch.cat(targets_merged, 0)


class CrowdDetectionCollateFN(DetectionCollateFN):
    """
    Collate function for Yolox training with additional_batch_items that includes crowd targets
    """

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch = default_collate(data)
        ims, targets, crowd_targets = batch[0:3]
        return ims, self._format_targets(targets), {"crowd_targets": self._format_targets(crowd_targets)}
