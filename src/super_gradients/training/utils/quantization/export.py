import torch
from torch.onnx import TrainingMode

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

try:
    from pytorch_quantization import nn as quant_nn

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


def export_quantized_module_to_onnx(model: torch.nn.Module, onnx_filename: str, input_shape: tuple, train: bool = False, **kwargs):
    """
    Method for exporting onnx after QAT.

    :param model: torch.nn.Module, model to export
    :param onnx_filename: str, target path for the onnx file,
    :param input_shape: tuple, input shape (usually BCHW)
    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    use_fb_fake_quant_state = quant_nn.TensorQuantizer.use_fb_fake_quant
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    logger.info("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)

    if train:
        training_mode = TrainingMode.TRAINING
        model.train()
    else:
        training_mode = TrainingMode.EVAL
        model.eval()
        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(**kwargs)

    torch.onnx.export(
        model, dummy_input, onnx_filename, verbose=False, opset_version=13, enable_onnx_checker=False, do_constant_folding=True, training=training_mode
    )

    # Restore functions of quant_nn back as expected
    quant_nn.TensorQuantizer.use_fb_fake_quant = use_fb_fake_quant_state
