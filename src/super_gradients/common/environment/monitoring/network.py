import psutil
from super_gradients.common.environment.monitoring.utils import bytes_to_megabytes

buffer_network_bytes_sent = psutil.net_io_counters().bytes_sent
buffer_network_bytes_recv = psutil.net_io_counters().bytes_recv


def get_network_sent_mb() -> float:
    """Number of MegaBytes sent since import"""
    return bytes_to_megabytes(psutil.net_io_counters().bytes_sent - buffer_network_bytes_sent)


def get_network_recv_mb() -> float:
    """Number of MegaBytes received since import"""
    return bytes_to_megabytes(psutil.net_io_counters().bytes_recv - buffer_network_bytes_recv)


def reset_network_sent():
    """Reset the value of net_io_counters"""
    global buffer_network_bytes_sent
    buffer_network_bytes_sent = psutil.net_io_counters().bytes_sent


def reset_network_recv():
    """Reset the value of net_io_counters"""
    global buffer_network_bytes_recv
    buffer_network_bytes_recv = psutil.net_io_counters().bytes_recv
