import psutil


def get_cpu_percent() -> float:
    """Average of all the CPU utilization."""
    return psutil.cpu_percent(interval=None, percpu=False)
