import psutil


def virtual_memory_used_percent():
    return psutil.virtual_memory().percent
