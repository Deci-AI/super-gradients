import dataclasses
from functools import partial
from typing import Callable, List, Iterator

from super_gradients.common.environment.monitoring.gpu import init_nvidia_management_lib, count_gpus


@dataclasses.dataclass
class StatAggregator:
    """Accumulate statistics samples and aggregates them.

    :param name:            Name of the statistic
    :param sampling_fn:     How the statistic is sampled
    :param aggregate_fn:    How the statistic samples are aggregated
    """

    name: str
    sampling_fn: Callable
    aggregate_fn: Callable
    _samples: List = dataclasses.field(default_factory=list)

    def sample(self):
        self._samples.append(self.sampling_fn())

    def aggregate(self) -> float:
        value = self.aggregate_fn(self._samples)
        self._samples = []
        return value


@dataclasses.dataclass
class GPUStatAggregatorIterator:
    """Iterator of multiple StatAggregator, that accumulate samples and aggregates them for each NVIDIA device.

    :param name:            Name of the statistic
    :param sampling_fn:     How the statistic is sampled
    :param aggregate_fn:    How the statistic samples are aggregated
    """

    name: str
    device_sampling_fn: Callable
    device_aggregate_fn: Callable
    _per_device_stat_aggregator: List[StatAggregator] = dataclasses.field(init=False)

    def __post_init__(self):
        """Initialize nvidia_management_lib and create a list of StatAggregator, one for each NVIDIA device."""
        init_nvidia_management_lib()
        self._per_device_stat_aggregator = [
            StatAggregator(name=f"{self.name}/device_{i}", sampling_fn=partial(self.device_sampling_fn, i), aggregate_fn=self.device_aggregate_fn)
            for i in range(count_gpus())
        ]

    def __iter__(self) -> Iterator[StatAggregator]:
        """Iterate over the StatAggregator of each node"""
        return iter(self._per_device_stat_aggregator)
