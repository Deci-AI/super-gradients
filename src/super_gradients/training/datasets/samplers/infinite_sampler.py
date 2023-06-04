from typing import Optional

from deprecated import deprecated
from torch.utils.data import DistributedSampler, Dataset
from super_gradients.common.registry.registry import register_sampler


@register_sampler()
class InfiniteSampler(DistributedSampler):
    @deprecated(version="3.2.0", reason="InfiniteSampler is deprecated and will be removed in 3.2.0. Using equivalent " "DistributedSampler.")
    def __init__(
        self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False
    ) -> None:
        super(InfiniteSampler, self).__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
