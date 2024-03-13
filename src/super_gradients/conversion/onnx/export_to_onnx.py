from functools import partial
from typing import Optional, Union, Mapping, Sequence

import torch
from torch import nn, Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.utils import check_model_contains_quantized_modules

logger = get_logger(__name__)


@torch.no_grad()
def export_to_onnx(
    *,
    model: nn.Module,
    model_input: Tensor,
    onnx_filename: str,
    input_names: Optional[Union[str, Sequence[str]]] = None,
    output_names: Optional[Union[str, Sequence[str]]] = None,
    onnx_opset=None,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]] = None,
    keep_initializers_as_inputs=None,
    verbose: bool = False,
) -> None:
    """
    Export model to ONNX format and save it to a file at `onnx_filename` location.

    A model can contain quantized modules. If it does, the export will be performed with fake quantization enabled.
    This function does not change model mode or device. User is responsible for setting model to eval mode and passing
    model_input argument with matching device.

    :param model: A model to convert to ONNX
    :param model_input: A sample input to the model
    :param onnx_filename: Path to save the ONNX file
    :param input_names: Names of the input tensors
    :param output_names: Names of the output tensors
    :param onnx_opset: ONNX opset version
    :param do_constant_folding: Whether to execute constant folding
    :param dynamic_axes: Dictionary of dynamic axes
    :param keep_initializers_as_inputs: Whether to keep initializers as inputs
    :param verbose: Whether to print verbose output
    :return: None
    """

    device = model_input.device
    for name, p in model.named_parameters():
        if p.device != device:
            logger.warning(f"Model parameter {name} is on device {p.device} but expected to be on device {device}")

    for name, p in model.named_buffers():
        if p.device != device:
            logger.warning(f"Model buffer {name} is on device {p.device} but expected to be on device {device}")

    # Sanity check that model works
    _ = model(model_input)

    logger.debug("Exporting model to ONNX")
    logger.debug(f"ONNX input shape: {model_input.shape} with dtype: {model_input.dtype}")
    logger.debug(f"ONNX output names: {output_names}")

    export_call = partial(
        torch.onnx.export,
        model=model,
        args=model_input,
        f=onnx_filename,
        input_names=input_names,
        output_names=output_names,
        opset_version=onnx_opset,
        do_constant_folding=do_constant_folding,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        verbose=verbose,
    )

    contains_quantized_modules = check_model_contains_quantized_modules(model)
    if contains_quantized_modules:
        from super_gradients.training.utils.quantization import use_fb_fake_quant

        with use_fb_fake_quant(True):
            export_call()
    else:
        export_call()
