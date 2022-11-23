import dataclasses
from functools import partial
from typing import Callable, List, Iterator, Dict

from super_gradients.common.environment.monitoring.gpu import init_nvidia_management_lib, count_gpus


@dataclasses.dataclass
class Scalar:
    name: str
    value: float


@dataclasses.dataclass
class Scalars:
    name: str
    values: Dict[str, float]


@dataclasses.dataclass
class StatAggregator:
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

    def aggregate_to_scalar(self) -> Scalar:
        return Scalar(self.name, self.aggregate())


@dataclasses.dataclass
class GPUStatAggregator:
    name: str
    device_sampling_fn: Callable
    device_aggregate_fn: Callable
    _per_device_stat_aggregator: List[StatAggregator] = dataclasses.field(init=False)

    def __post_init__(self):
        init_nvidia_management_lib()
        self._per_device_stat_aggregator = [
            StatAggregator(name=f"{self.name}/device_{i}", sampling_fn=partial(self.device_sampling_fn, i), aggregate_fn=self.device_aggregate_fn)
            for i in range(count_gpus())
        ]

    def __iter__(self) -> Iterator[StatAggregator]:
        return iter(self._per_device_stat_aggregator)

    def sample(self):
        for stat_aggregator in self._per_device_stat_aggregator:
            stat_aggregator.sample()

    def aggregate(self) -> Dict[str, float]:
        return {f"device_{i}": stat_aggregator.aggregate() for i, stat_aggregator in enumerate(self._per_device_stat_aggregator)}

    def aggregate_to_scalar(self) -> Scalars:
        return Scalars(name=self.name, values=self.aggregate())
