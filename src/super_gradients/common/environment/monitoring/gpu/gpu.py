from super_gradients.common.environment.monitoring.gpu import pynvml


def init_nvidia_management_lib():
    """Initialize nvml (NVDIA management library), which is required to use pynvml."""
    pynvml.nvmlInit()


def count_gpus() -> int:
    """Count how many GPUS NVDIA detects."""
    return pynvml.nvmlDeviceGetCount()


def get_device_memory_usage_percent(gpu_index: int) -> float:
    """GPU memory utilization in percent of a given GPU."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).memory


def get_device_memory_allocated_percent(gpu_index: int) -> float:
    """GPU memory allocated in percent of a given GPU."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return memory_info.used / memory_info.total * 100


def get_device_usage_percent(gpu_index: int) -> float:
    """GPU utilization in percent of a given GPU."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


def get_device_temperature_c(gpu_index: int) -> float:
    """GPU temperature in Celsius of a given GPU."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)


def get_device_power_usage_w(gpu_index: int) -> float:
    """GPU power usage in Watts of a given GPU."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Wats


def get_device_power_usage_percent(gpu_index: int) -> float:
    """GPU power usage in percent of a given GPU."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    power_watts = pynvml.nvmlDeviceGetPowerUsage(handle)
    power_capacity_watts = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
    return (power_watts / power_capacity_watts) * 100
