import psutil


def virtual_memory_used_percent() -> float:
    """Virtual memory used in percent."""
    return psutil.virtual_memory().percent
