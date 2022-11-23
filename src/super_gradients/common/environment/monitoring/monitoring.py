import time
import threading
from typing import List, Union

from super_gradients.common.sg_loggers import BaseSGLogger
from super_gradients.common.environment.env_helpers import multi_process_safe
from super_gradients.common.environment.monitoring.data_models import StatAggregator, GPUStatAggregator, Scalar, Scalars
from super_gradients.common.environment.monitoring.cpu import get_disk_usage_percent, get_io_read_mbs, get_io_write_mbs
from super_gradients.common.environment.monitoring.gpu import (
    get_device_memory_utilization,
    get_device_memory_allocated_percent,
    get_device_utilization,
    get_device_temperature,
    get_device_power_usage,
    get_device_power_usage_percent,
)
from super_gradients.common.environment.monitoring.utils import average

import psutil


def get_cpu_percent():
    return psutil.cpu_percent(interval=None, percpu=False)


initial_net_io_counters = psutil.net_io_counters()


def get_network_sent():
    return psutil.net_io_counters().bytes_sent - initial_net_io_counters.bytes_sent


def get_network_recv():
    return psutil.net_io_counters().bytes_recv - initial_net_io_counters.bytes_recv


def virtual_memory_used_percent():
    return psutil.virtual_memory().percent


class SystemMonitor:
    def __init__(self, sg_logger: BaseSGLogger):
        self.sg_logger = sg_logger

        self.stats_aggregators: List[StatAggregator] = [
            StatAggregator(name="System/disk.usage_percent", sampling_fn=get_disk_usage_percent, aggregate_fn=average),
            StatAggregator(name="System/disk.io_read_mbs", sampling_fn=get_io_read_mbs, aggregate_fn=average),
            StatAggregator(name="System/disk.io_write_mbs", sampling_fn=get_io_write_mbs, aggregate_fn=average),
            StatAggregator(name="System/memory.usage_percent", sampling_fn=virtual_memory_used_percent, aggregate_fn=average),
            StatAggregator(name="System/network.network_sent", sampling_fn=get_network_sent, aggregate_fn=average),
            StatAggregator(name="System/network.network_recv", sampling_fn=get_network_recv, aggregate_fn=average),
            *GPUStatAggregator(name="System/gpu.usage_percent", device_sampling_fn=get_device_memory_utilization, device_aggregate_fn=average),
            *GPUStatAggregator(name="System/gpu.memory_allocated_percent", device_sampling_fn=get_device_memory_allocated_percent, device_aggregate_fn=average),
            *GPUStatAggregator(name="System/gpu.utilization", device_sampling_fn=get_device_utilization, device_aggregate_fn=average),
            *GPUStatAggregator(name="System/gpu.temperature", device_sampling_fn=get_device_temperature, device_aggregate_fn=average),
            *GPUStatAggregator(name="System/gpu.power_usage", device_sampling_fn=get_device_power_usage, device_aggregate_fn=average),
            *GPUStatAggregator(name="System/gpu.power_usage_percent", device_sampling_fn=get_device_power_usage_percent, device_aggregate_fn=average),
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
        """Sample the stats_aggregators, i.e. get the current value of each of them."""
        for stat in self.stats_aggregators:
            stat.sample()

    def _aggregate_and_write(self):
        """Aggregate and write the results."""
        self.count += 1
        for stat in self.stats_aggregators:
            self._write_scalar(scalar=stat.aggregate_to_scalar())

    def _write_scalar(self, scalar: Union[Scalar, Scalars]):
        """Write a scalar with sg_logger (can be written on Tensorboard, WandB, ...)"""
        if isinstance(scalar, Scalar):
            self.sg_logger.add_scalar(tag=scalar.name, scalar_value=scalar.value, global_step=self.count)
        else:
            self.sg_logger.add_scalars_to_same_plot(tag=scalar.name, tag_scalar_dict=scalar.values, global_step=self.count)

    @classmethod
    @multi_process_safe
    def start(cls, sg_logger: BaseSGLogger):
        """Instantiate a SystemMonitor in a multiprocess safe way."""
        return cls.__init__(sg_logger=sg_logger)
