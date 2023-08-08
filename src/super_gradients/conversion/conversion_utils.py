import numpy as np
import torch

__all__ = ["torch_dtype_to_numpy_dtype", "numpy_dtype_to_torch_dtype"]

_DTYPE_CORRESPONDENCE = [
    (torch.float32, np.float32),
    (torch.float64, np.float64),
    (torch.float16, np.float16),
    (torch.int32, np.int32),
    (torch.int64, np.int64),
    (torch.int16, np.int16),
    (torch.int8, np.int8),
    (torch.uint8, np.uint8),
    (torch.bool, np.bool),
]


def torch_dtype_to_numpy_dtype(dtype: torch.dtype):
    for torch_dtype, numpy_dtype in _DTYPE_CORRESPONDENCE:
        if dtype == torch_dtype:
            return numpy_dtype
    raise NotImplementedError(f"Unsupported dtype: {dtype}")


def numpy_dtype_to_torch_dtype(dtype: np.dtype):
    for torch_dtype, numpy_dtype in _DTYPE_CORRESPONDENCE:
        if dtype == numpy_dtype:
            return torch_dtype
    raise NotImplementedError(f"Unsupported dtype: {dtype}")
