from super_gradients.common.environment.monitoring.gpu import pynvml


def init_nvidia_management_lib():
    pynvml.nvmlInit()


def count_gpus() -> int:
    return pynvml.nvmlDeviceGetCount()


def get_device_memory_utilization(i: int) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).memory


def get_device_memory_allocated_percent(i: int) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return memory_info.used / memory_info.total * 100


def get_device_utilization(i: int) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


def get_device_temperature(i: int) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)


def get_device_power_usage(i: int) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Wats


def get_device_power_usage_percent(i: int) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    power_watts = pynvml.nvmlDeviceGetPowerUsage(handle)
    power_capacity_watts = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
    return (power_watts / power_capacity_watts) * 100
