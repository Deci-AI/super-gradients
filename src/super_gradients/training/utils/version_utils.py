import torch

_TORCH_VERSION_MAJOR_MINOR_ = tuple(map(int, torch.version.__version__.split(".")[:2]))

__all__ = ["torch_version_is_greater_or_equal"]


def torch_version_is_greater_or_equal(major: int, minor: int) -> bool:
    version = (major, minor)
    return _TORCH_VERSION_MAJOR_MINOR_ >= version
