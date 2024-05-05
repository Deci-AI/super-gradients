import collections
from typing import Union, Dict, List
import torch
from torch import Tensor
from torchmetrics import Metric

import super_gradients
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.ddp_utils import get_world_size
from super_gradients.common.registry import register_metric
from super_gradients.common.object_names import Metrics

logger = get_logger(__name__)

__all__ = ["EPE"]


@register_metric(Metrics.EPE)
class EPE(Metric):
    """
    End-Point-Error metric for optical flow.

    :param max_flow: The maximum flow displacement allowed. Flow values above it will be excluded from metric calculation.
    """

    def __init__(self, max_flow: int):
        super().__init__()

        greater_component_is_better = [
            ("epe", False),
        ]

        self.max_flow = max_flow
        self.greater_component_is_better = collections.OrderedDict(greater_component_is_better)
        self.component_names = list(self.greater_component_is_better.keys())
        self.components = len(self.component_names)
        self.world_size = None
        self.rank = None
        self.is_distributed = super_gradients.is_distributed()

        self.add_state("epe", default=[], dist_reduce_fx="cat")

    def update(self, preds: List[Tensor], target: Tensor):
        flow_gt, valid = target

        if torch.is_tensor(preds):
            preds = [preds]

        # exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        epe = torch.sum((preds[-1] - flow_gt) ** 2, dim=1).sqrt()

        epe = epe.view(-1)[valid.view(-1)]
        epe = epe.mean().item()

        self.epe.append(torch.tensor(epe, dtype=torch.float32))

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        return dict(epe=torch.tensor(self.epe).mean().item())

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        """
        When in distributed mode, stats are aggregated after each forward pass to the metric state. Since these have all
        different sizes we override the synchronization function since it works only for tensors (and use
        all_gather_object)
        """
        if self.world_size is None:
            self.world_size = get_world_size() if self.is_distributed else -1
        if self.rank is None:
            self.rank = torch.distributed.get_rank() if self.is_distributed else -1

        if self.is_distributed:
            local_state_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
            gathered_state_dicts = [None] * self.world_size
            torch.distributed.barrier()
            torch.distributed.all_gather_object(gathered_state_dicts, local_state_dict)
            metric_keys = {"epe": []}

            for state_dict in gathered_state_dicts:
                for key in state_dict.keys():
                    if len(state_dict[key]) > 0:
                        metric_keys[key].extend(state_dict[key])

            for key in metric_keys.keys():
                setattr(self, key, metric_keys[key])
