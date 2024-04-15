import collections
from typing import Union, Dict, List
import torch
from torch import Tensor
from torchmetrics import Metric

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry import register_metric
from super_gradients.common.object_names import Metrics

logger = get_logger(__name__)

__all__ = ["EPE"]


@register_metric(Metrics.EPE)
class EPE(Metric):
    """
    End-Point-Error metric for optical flow.

    :param max_flow: The maximum flow displacement allowed. Flow values above it will be excluded from metric calculation.
    :param dist_sync_on_step: Synchronize metric state across processes at each ``forward()`` before returning the value at the step.
    """

    def __init__(self, max_flow: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        greater_component_is_better = [
            ("epe", False),
        ]

        self.max_flow = max_flow
        self.greater_component_is_better = collections.OrderedDict(greater_component_is_better)
        self.component_names = list(self.greater_component_is_better.keys())
        self.components = len(self.component_names)

        self.add_state("epe", default=[], dist_reduce_fx="cat")

    def update(self, preds: List[Tensor], target: Tensor):
        flow_gt, valid = target

        # exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        epe = torch.sum((preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
        epe = epe.mean().item()
        self.epe.append(torch.tensor(epe, dtype=torch.float32))

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        return dict(epe=torch.tensor(self.epe).mean().item())
