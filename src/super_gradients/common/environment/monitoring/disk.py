import psutil

from super_gradients.common.environment.monitoring.utils import bytes_to_megabytes

buffer_io_read_bytes = psutil.disk_io_counters().read_bytes
buffer_io_write_bytes = psutil.disk_io_counters().write_bytes


def get_disk_usage_percent() -> float:
    """Disk memory used in percent."""
    return psutil.disk_usage("/").percent


def get_io_read_mb() -> float:
    """Number of MegaBytes read since import"""
    return bytes_to_megabytes(psutil.disk_io_counters().read_bytes - buffer_io_read_bytes)


def get_io_write_mb() -> float:
    """Number of MegaBytes written since import"""
    return bytes_to_megabytes(psutil.disk_io_counters().write_bytes - buffer_io_write_bytes)


def reset_io_read():
    """Reset the value of net_io_counters"""
    global buffer_io_read_bytes
    buffer_io_read_bytes = psutil.disk_io_counters().read_bytes


def reset_io_write():
    """Reset the value of net_io_counters"""
    global buffer_io_write_bytes
    buffer_io_write_bytes = psutil.disk_io_counters().write_bytes
