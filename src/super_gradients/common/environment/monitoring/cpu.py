import psutil


def get_cpu_percent():
    return psutil.cpu_percent(interval=None, percpu=False)
