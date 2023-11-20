import numpy as np
import torch

__all__ = ["torch_dtype_to_numpy_dtype", "numpy_dtype_to_torch_dtype", "find_compatible_model_device_for_dtype"]

_DTYPE_CORRESPONDENCE = [
    (torch.float32, np.float32),
    (torch.float64, np.float64),
    (torch.float16, np.float16),
    (torch.int32, np.int32),
    (torch.int64, np.int64),
    (torch.int16, np.int16),
    (torch.int8, np.int8),
    (torch.uint8, np.uint8),
    (torch.bool, bool),
]


def find_compatible_model_device_for_dtype(device: torch.device, dtype: torch.dtype):
    """
    Helper method to handle lack of FP16+CPU support in PyTorch.
    Pytorch does not support exporting to ONNX model with weights in FP16 on CPU yet.
    Therefore, we should do the export using CUDA if it's available

    :param device: A current model's device
    :param dtype:  A current model's dtype of weights/inputs
    """
    if dtype == torch.half:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError(
                "Pytorch does not support exporting to ONNX model with weights in FP16 on CPU/MPS yet."
                "This is not a bug of SuperGradients, rather lack of fp16+cpu routines in PyTorch."
                "No CUDA device detected which means you can't export model with FP16 quantization on this machine."
            )

    return device


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
