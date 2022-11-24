import psutil
from super_gradients.common.environment.monitoring.utils import bytes_to_megabytes

initial_net_io_counters = psutil.net_io_counters()


def get_network_sent_mb() -> float:
    """Number of MegaBytes sent since import"""
    return bytes_to_megabytes(psutil.net_io_counters().bytes_sent - initial_net_io_counters.bytes_sent)


def get_network_recv_mb() -> float:
    """Number of MegaBytes received since import"""
    return bytes_to_megabytes(psutil.net_io_counters().bytes_recv - initial_net_io_counters.bytes_recv)
