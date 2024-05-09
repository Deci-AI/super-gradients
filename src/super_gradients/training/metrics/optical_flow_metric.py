import collections
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Tuple
import torch
from torchmetrics import Metric

import super_gradients
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.ddp_utils import get_world_size
from super_gradients.common.registry import register_metric
from super_gradients.common.object_names import Metrics

logger = get_logger(__name__)

__all__ = ["EPE"]


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


class PreprocessOpticalFlowMetricsArgs(AbstractMetricsArgsPrepFn):
    """
    Default optical flow inputs preprocess function before updating optical flow metrics, handles multiple inputs.
    """

    def __init__(self, pad_factor: int = 8, apply_unpad: bool = False):
        """
        :param pad_factor: The factor by which the input images were padded. By default, set to 8.
        :param apply_unpad: Whether to apply unpading on predictions list. By default, set to False.
        """
        self.pad_factor = pad_factor
        self.apply_unpad = apply_unpad

    def __call__(self, preds, target: torch.Tensor) -> List[torch.Tensor]:
        # WHEN DEALING WITH MULTIPLE OUTPUTS- OUTPUTS[-1] IS THE MAIN FLOW MAP
        ht, wd = target.shape[-2:]
        pad_ht = (((ht // self.pad_factor) + 1) * self.pad_factor - ht) % self.pad_factor
        pad_wd = (((wd // self.pad_factor) + 1) * self.pad_factor - wd) % self.pad_factor
        pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        if self.apply_unpad:
            ht, wd = preds.shape[-2:]
            c = [pad[2], ht - pad[3], pad[0], wd - pad[1]]
            preds = preds[..., c[0] : c[1], c[2] : c[3]]

        return [preds]


@register_metric(Metrics.EPE)
class EPE(Metric):
    """
    End-Point-Error metric for optical flow.

    :param max_flow: The maximum flow displacement allowed. Flow values above it will be excluded from metric calculation.
    :param apply_unpad: Bool, if to apply unpad to the predicted flow map. By default, set to False.
    """

    def __init__(self, max_flow: int = None, apply_unpad: bool = False):
        super().__init__()

        greater_component_is_better = [
            ("epe", False),
        ]

        self.max_flow = max_flow
        self.metrics_args_prep_fn = PreprocessOpticalFlowMetricsArgs(apply_unpad=apply_unpad)
        self.greater_component_is_better = collections.OrderedDict(greater_component_is_better)
        self.component_names = list(self.greater_component_is_better.keys())
        self.components = len(self.component_names)
        self.world_size = None
        self.rank = None
        self.is_distributed = super_gradients.is_distributed()

        self.add_state("epe", default=[], dist_reduce_fx="cat")

    def update(self, preds: List[torch.Tensor], target: torch.Tensor):
        flow_gt, valid = target

        if torch.is_tensor(preds):
            preds = [preds]

        preds = self.metrics_args_prep_fn(preds, flow_gt)

        # exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        epe = torch.sum((preds[-1] - flow_gt) ** 2, dim=1).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        valid = valid.view(-1)

        if self.max_flow is None:
            valid = valid >= 0.5
        else:
            valid = (valid >= 0.5) & (mag < self.max_flow)

        epe = epe[valid].mean().item()

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
