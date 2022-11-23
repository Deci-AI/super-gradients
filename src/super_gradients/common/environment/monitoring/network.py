import psutil


initial_net_io_counters = psutil.net_io_counters()


def get_network_sent():
    return psutil.net_io_counters().bytes_sent - initial_net_io_counters.bytes_sent


def get_network_recv():
    return psutil.net_io_counters().bytes_recv - initial_net_io_counters.bytes_recv
