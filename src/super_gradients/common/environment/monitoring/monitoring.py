import time
import threading
from typing import List, Union

from super_gradients.common.sg_loggers import BaseSGLogger
from super_gradients.common.environment.env_helpers import multi_process_safe
from super_gradients.common.environment.monitoring.data_models import Metric, GPUMetricIterator, Scalar, Scalars
from super_gradients.common.environment.monitoring.cpu import get_disk_usage_percent, get_io_read_mbs, get_io_write_mbs
from super_gradients.common.environment.monitoring.gpu import get_device_memory_utilization
from super_gradients.common.environment.monitoring.utils import average


class MetricsHandler:
    def __init__(self, sg_logger: BaseSGLogger):
        self.sg_logger = sg_logger

        self.metrics: List[Metric] = [
            Metric(name="system/disk.usage_percent", sampling_fn=get_disk_usage_percent, aggregate_fn=average),
            Metric(name="system/disk.io_read_mbs", sampling_fn=get_io_read_mbs, aggregate_fn=average),
            Metric(name="system/disk.io_write_mbs", sampling_fn=get_io_write_mbs, aggregate_fn=average),
            GPUMetricIterator(name="system/gpu.usage_percent", device_sampling_fn=get_device_memory_utilization, device_aggregate_fn=average),
        ]

        self.count = 0

        self.aggregate_frequency = 10  # in sec
        self.n_samples_per_aggregate = 100
        self.sample_interval = self.aggregate_frequency / self.n_samples_per_aggregate

        thread = threading.Thread(target=self._run, daemon=True, name="SystemMonitor")
        thread.start()

    def _run(self):
        """Run in the background"""
        while True:
            for _ in range(self.n_samples_per_aggregate):
                self._sample()
                time.sleep(self.sample_interval)
            self._aggregate_and_write()

    def _sample(self):
        """Sample the metrics, i.e. get the current value of each of them."""
        for metric in self.metrics:
            metric.sample()

    def _aggregate_and_write(self):
        """Aggregate and write the results."""
        self.count += 1
        for metric in self.metrics:
            self._write_scalar(scalar=metric.aggregate_to_scalar())

    def _write_scalar(self, scalar: Union[Scalar, Scalars]):
        """Write a scalar with sg_logger (can be written on Tensorboard, WandB, ...)"""
        if isinstance(scalar, Scalar):
            self.sg_logger.add_scalar(tag=scalar.name, scalar_value=scalar.value, global_step=self.count)
        else:
            self.sg_logger.add_scalars_to_same_plot(tag=scalar.name, tag_scalar_dict=scalar.values, global_step=self.count)

    @classmethod
    @multi_process_safe
    def start(cls, sg_logger: BaseSGLogger):
        """Instantiate a MetricsHandler in a multiprocess safe way."""
        return cls.__init__(sg_logger=sg_logger)
