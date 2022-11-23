from super_gradients.common.environment.monitoring.gpu import pynvml


def count_gpus() -> int:
    return pynvml.nvmlDeviceGetCount()


def get_device_memory_utilization(i: int) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).memory
