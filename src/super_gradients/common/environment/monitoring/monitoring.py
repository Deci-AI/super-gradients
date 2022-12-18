import time
import threading

from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.environment.monitoring import disk, virtual_memory, network, cpu, gpu
from super_gradients.common.environment.monitoring.utils import average, delta_per_s
from super_gradients.common.environment.monitoring.data_models import StatAggregator, GPUStatAggregatorIterator
from torch.utils.tensorboard import SummaryWriter


class SystemMonitor:
    """Monitor and write to tensorboard the system statistics, such as CPU usage, GPU, ...

    :param tensorboard_writer:  Tensorboard object that will be used to save the statistics
    :param extra_gpu_stats:     Set to True to get extra gpu statistics, such as gpu temperature, power usage, ...
                                Default set to False, because this reduces the tensorboard readability.
    """

    def __init__(self, tensorboard_writer: SummaryWriter, extra_gpu_stats: bool = False):
        self.tensorboard_writer = tensorboard_writer
        self.write_count = 0
        self.running = True

        self.aggregate_frequency = 30  # in sec
        self.n_samples_per_aggregate = 60
        self.sample_interval = self.aggregate_frequency / self.n_samples_per_aggregate

        self.stat_aggregators = [
            StatAggregator(name="System/disk.usage_percent", sampling_fn=disk.get_disk_usage_percent, aggregate_fn=average),
            StatAggregator(name="System/disk.io_write_mbs", sampling_fn=disk.get_io_write_mb, aggregate_fn=delta_per_s, reset_callback_fn=disk.reset_io_write),
            StatAggregator(name="System/disk.io_read_mbs", sampling_fn=disk.get_io_read_mb, aggregate_fn=delta_per_s, reset_callback_fn=disk.reset_io_read),
            StatAggregator(name="System/memory.usage_percent", sampling_fn=virtual_memory.virtual_memory_used_percent, aggregate_fn=average),
            StatAggregator(
                name="System/network.network_sent_mbs",
                sampling_fn=network.get_network_sent_mb,
                aggregate_fn=delta_per_s,
                reset_callback_fn=network.reset_network_sent,
            ),
            StatAggregator(
                name="System/network.network_recv_mbs",
                sampling_fn=network.get_network_recv_mb,
                aggregate_fn=delta_per_s,
                reset_callback_fn=network.reset_network_recv,
            ),
            StatAggregator(name="System/cpu.usage_percent", sampling_fn=cpu.get_cpu_percent, aggregate_fn=average),
        ]

        is_nvidia_lib_available = gpu.safe_init_nvidia_management_lib()
        if is_nvidia_lib_available:
            self.stat_aggregators += [
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
        """Sample, aggregate and write the statistics regularly."""
        self._init_stat_aggregators()
        while self.running:
            for _ in range(self.n_samples_per_aggregate):
                self._sample()
                time.sleep(self.sample_interval)
                if not self.running:
                    break
            self._aggregate_and_write()

    def _init_stat_aggregators(self):
        for stat_aggregator in self.stat_aggregators:
            stat_aggregator.reset()

    def _sample(self):
        """Sample the stat_aggregators, i.e. get the current value of each of them."""
        for stat_aggregator in self.stat_aggregators:
            stat_aggregator.sample()

    def _aggregate_and_write(self):
        """Aggregate and write the results."""
        for stat_aggregator in self.stat_aggregators:
            scalar = stat_aggregator.aggregate()
            if scalar is not None:
                self.tensorboard_writer.add_scalar(tag=stat_aggregator.name, scalar_value=scalar, global_step=self.write_count)
            stat_aggregator.reset()
        self.write_count += 1

    @classmethod
    @multi_process_safe
    def start(cls, tensorboard_writer: SummaryWriter):
        """Instantiate a SystemMonitor in a multiprocess safe way."""
        return cls(tensorboard_writer=tensorboard_writer)

    def close(self):
        self.running = False
