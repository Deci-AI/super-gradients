from typing import List


def average(lst: List[float]):
    return sum(lst) / len(lst) if lst else None


def bytes_to_megabytes(x):
    BYTES_PER_MEGABYTE = 1024**2
    return x / BYTES_PER_MEGABYTE
