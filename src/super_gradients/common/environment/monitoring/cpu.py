import psutil

from super_gradients.common.environment.monitoring.utils import bytes_to_megabytes


def get_disk_usage_percent() -> float:
    return psutil.disk_usage("/").percent


def get_io_read_mbs() -> float:
    return bytes_to_megabytes(psutil.disk_io_counters().read_bytes)


def get_io_write_mbs() -> float:
    return bytes_to_megabytes(psutil.disk_io_counters().write_bytes)
