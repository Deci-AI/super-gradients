import time
import threading

from super_gradients.common.environment.env_helpers import multi_process_safe
from super_gradients.common.environment.monitoring import disk, virtual_memory, network, cpu, gpu
from super_gradients.common.environment.monitoring.utils import average
from super_gradients.common.environment.monitoring.data_models import StatAggregator, GPUStatAggregatorIterator
from torch.utils.tensorboard import SummaryWriter


class SystemMonitor:
    def __init__(self, tensorboard_writer: SummaryWriter, extra_gpu_stats: bool = False):
        self.tensorboard_writer = tensorboard_writer
        self.write_count = 0

        self.aggregate_frequency = 10  # in sec
        self.n_samples_per_aggregate = 100
        self.sample_interval = self.aggregate_frequency / self.n_samples_per_aggregate

        self.stat_aggregators = [
            StatAggregator(name="System/disk.usage_percent", sampling_fn=disk.get_disk_usage_percent, aggregate_fn=average),
            StatAggregator(name="System/disk.io_read_mbs", sampling_fn=disk.get_io_read_mbs, aggregate_fn=average),
            StatAggregator(name="System/disk.io_write_mbs", sampling_fn=disk.get_io_write_mbs, aggregate_fn=average),
            StatAggregator(name="System/memory.usage_percent", sampling_fn=virtual_memory.virtual_memory_used_percent, aggregate_fn=average),
            StatAggregator(name="System/network.network_sent_mb", sampling_fn=network.get_network_sent_mb, aggregate_fn=average),
            StatAggregator(name="System/network.network_recv_mb", sampling_fn=network.get_network_recv_mb, aggregate_fn=average),
            StatAggregator(name="System/cpu.usage_percent", sampling_fn=cpu.get_cpu_percent, aggregate_fn=average),
            *GPUStatAggregatorIterator(
                name="System/gpu.memory_usage_percent", device_sampling_fn=gpu.get_device_memory_usage_percent, device_aggregate_fn=average
            ),
            *GPUStatAggregatorIterator(
                name="System/gpu.memory_allocated_percent", device_sampling_fn=gpu.get_device_memory_allocated_percent, device_aggregate_fn=average
            ),
            *GPUStatAggregatorIterator(name="System/gpu.usage_percent", device_sampling_fn=gpu.get_device_usage_percent, device_aggregate_fn=average),
        ]

        if extra_gpu_stats:
            self.stat_aggregators += [
                *GPUStatAggregatorIterator(name="System/gpu.temperature_c", device_sampling_fn=gpu.get_device_temperature_c, device_aggregate_fn=average),
                *GPUStatAggregatorIterator(name="System/gpu.power_usage_w", device_sampling_fn=gpu.get_device_power_usage_w, device_aggregate_fn=average),
                *GPUStatAggregatorIterator(
                    name="System/gpu.power_usage_percent", device_sampling_fn=gpu.get_device_power_usage_percent, device_aggregate_fn=average
                ),
            ]

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
        """Sample the stat_aggregators, i.e. get the current value of each of them."""
        for stat_aggregator in self.stat_aggregators:
            stat_aggregator.sample()

    def _aggregate_and_write(self):
        """Aggregate and write the results."""
        for stat_aggregator in self.stat_aggregators:
            self.tensorboard_writer.add_scalar(tag=stat_aggregator.name, scalar_value=stat_aggregator.aggregate(), global_step=self.write_count)
        self.write_count += 1

    @classmethod
    @multi_process_safe
    def start(cls, tensorboard_writer: SummaryWriter):
        """Instantiate a SystemMonitor in a multiprocess safe way."""
        cls(tensorboard_writer=tensorboard_writer)
