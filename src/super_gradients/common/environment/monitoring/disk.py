import psutil

from super_gradients.common.environment.monitoring.utils import bytes_to_megabytes

initial_net_io_counters = psutil.disk_io_counters()


def get_disk_usage_percent() -> float:
    """Disk memory used in percent."""
    return psutil.disk_usage("/").percent


def get_io_read_mb() -> float:
    """Number of MegaBytes read since import"""
    return bytes_to_megabytes(psutil.disk_io_counters().read_bytes - initial_net_io_counters.read_bytes)


def get_io_write_mb() -> float:
    """Number of MegaBytes written since import"""
    return bytes_to_megabytes(psutil.disk_io_counters().write_bytes - initial_net_io_counters.write_bytes)
