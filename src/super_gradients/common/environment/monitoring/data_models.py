import dataclasses
from functools import partial
from typing import Callable, List, Iterator, Dict

from super_gradients.common.environment.monitoring.gpu import pynvml, count_gpus


@dataclasses.dataclass
class Scalar:
    name: str
    value: float


@dataclasses.dataclass
class Scalars:
    name: str
    values: Dict[str, float]


@dataclasses.dataclass
class Metric:
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
class GPUMetricIterator:
    name: str
    device_sampling_fn: Callable
    device_aggregate_fn: Callable
    _per_device_metric: List[Metric] = dataclasses.field(init=False)

    def __post_init__(self):
        pynvml.nvmlInit()
        # self._per_device_metric = [
        #     Metric(name=f"{self.name}/device_{i}", sampling_fn=partial(self.device_sampling_fn, i), aggregate_fn=self.device_aggregate_fn)
        #     for i in range(count_gpus())
        # ]
        self._per_device_metric = [
            Metric(name=self.name, sampling_fn=partial(self.device_sampling_fn, i), aggregate_fn=self.device_aggregate_fn) for i in range(count_gpus())
        ]

    def __iter__(self) -> Iterator[Metric]:
        return iter(self._per_device_metric)

    def sample(self):
        for metric in self._per_device_metric:
            metric.sample()

    def aggregate(self) -> Dict[str, float]:
        return {f"device_{i}": metric.aggregate() for i, metric in enumerate(self._per_device_metric)}

    def aggregate_to_scalar(self) -> Scalars:
        return Scalars(name=self.name, values=self.aggregate())
        # return [{"name": f"device_{device_i}", "scalar": metric.aggregate()} for device_i, metric in enumerate(self._per_device_metric)]
        # return [{"name": f"device_{device_i}", "scalar": metric.aggregate()} for device_i, metric in enumerate(self._per_device_metric)]
