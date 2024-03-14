import torch
from torch.onnx import TrainingMode
from copy import deepcopy
from pytorch_quantization import nn as quant_nn

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.deprecate import deprecated
from super_gradients.conversion.onnx.export_to_onnx import export_to_onnx

logger = get_logger(__name__)


@deprecated(
    deprecated_since="3.7.0",
    removed_from="4.0.0",
    target=export_to_onnx,
)
def export_quantized_module_to_onnx(
    model: torch.nn.Module, onnx_filename: str, input_shape: tuple, train: bool = False, to_cpu: bool = True, deepcopy_model=False, **kwargs
):
    """
    Method for exporting onnx after QAT.

    :param to_cpu: transfer model to CPU before converting to ONNX, dirty workaround when model's tensors are on different devices
    :param train: export model in training mode
    :param model: torch.nn.Module, model to export
    :param onnx_filename: str, target path for the onnx file,
    :param input_shape: tuple, input shape (usually BCHW)
    :param deepcopy_model: Whether to export deepcopy(model). Necessary in case further training is performed and
     prep_model_for_conversion makes the network un-trainable (i.e RepVGG blocks).
    """
    if deepcopy_model:
        model = deepcopy(model)

    use_fb_fake_quant_state = quant_nn.TensorQuantizer.use_fb_fake_quant
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    logger.info("Creating ONNX file: " + onnx_filename)

    if train:
        training_mode = TrainingMode.TRAINING
        model.train()
    else:
        training_mode = TrainingMode.EVAL
        model.eval()
        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(**kwargs)

    # workaround when model.prep_model_for_conversion does reparametrization
    # and tensors get scattered to different devices
    if to_cpu:
        export_model = model.cpu()
    else:
        export_model = model

    dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
    torch.onnx.export(export_model, dummy_input, onnx_filename, verbose=False, opset_version=13, do_constant_folding=True, training=training_mode)

    # Restore functions of quant_nn back as expected
    quant_nn.TensorQuantizer.use_fb_fake_quant = use_fb_fake_quant_state
