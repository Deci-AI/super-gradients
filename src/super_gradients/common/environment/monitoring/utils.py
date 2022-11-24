from typing import List, Union


def average(lst: List[float]) -> Union[float, None]:
    """Average a list of values, return None if empty list"""
    return sum(lst) / len(lst) if lst else None


def bytes_to_megabytes(b: float) -> float:
    """Convert bytes to megabytes"""
    BYTES_PER_MEGABYTE = 1024**2
    return b / BYTES_PER_MEGABYTE
