from typing import List, Union


def average(samples: List[float], time_diff: float) -> Union[float, None]:
    """Average a list of values, return None if empty list"""
    return sum(samples) / len(samples) if samples else None


def delta_per_s(samples: List[float], time_diff: float) -> Union[float, None]:
    """Compute the difference per second (ex. megabytes per second), return None if empty list"""
    return (samples[-1] - samples[0]) / time_diff if samples else None


def bytes_to_megabytes(b: float) -> float:
    """Convert bytes to megabytes"""
    BYTES_PER_MEGABYTE = 1024**2
    return b / BYTES_PER_MEGABYTE
