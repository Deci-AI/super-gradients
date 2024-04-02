import copy
from functools import partial
from typing import Optional

from torch.utils.data import DataLoader

from .calibrator import QuantizationCalibrator
from .selective_quantization_utils import SelectiveQuantizer
from super_gradients.training.utils.utils import check_model_contains_quantized_modules
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def openvino_ptq_from_onnx(
    model,
    calibration_loader,
    quantization_skip_layers,
    calibration_batches: int,
    validation_loader: Optional[DataLoader] = None,
    validation_fn: Optional[None] = None,
):
    import nncf

    # import openvino as ov

    # if isinstance(model, str):
    #    onnx_model = onnx.load(model)
    #    input_name = onnx_model.graph.input[0].name
    # else:
    #    input_name = model.graph.input[0].name

    def transform_fn_to_numpy(data_item):
        images = data_item[0]
        return images.numpy()[0:1, ...]  # Batch size from calibration loader should match the batch size of the model

    ignored_scope = None

    if quantization_skip_layers is not None:
        ignored_scope = nncf.IgnoredScope(patterns=list(quantization_skip_layers))
        logger.debug(f"Quantization skip layers: {quantization_skip_layers}")

    if validation_loader is not None and validation_fn is not None:
        logger.debug("Starting model quantization using NNCF with QC")

        calibration_dataset = nncf.Dataset(calibration_loader, transform_func=transform_fn_to_numpy)
        validation_dataset = nncf.Dataset(validation_loader, transform_func=transform_fn_to_numpy)

        quantized_model = nncf.quantize_with_accuracy_control(
            model,
            calibration_dataset=calibration_dataset,
            validation_dataset=validation_dataset,
            validation_fn=validation_fn,
            ignored_scope=ignored_scope,
            subset_size=calibration_batches,
        )
        logger.debug("Model quantization using NNCF with QC completed")
    else:
        logger.debug("Starting model quantization using NNCF without QC")
        calibration_dataset = nncf.Dataset(calibration_loader, transform_func=transform_fn_to_numpy)
        quantized_model = nncf.quantize(
            model,
            calibration_dataset=calibration_dataset,
            ignored_scope=ignored_scope,
            subset_size=calibration_batches,  # TODO: Check whether subset_size is sample size or batch size
            preset=nncf.QuantizationPreset.MIXED,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                # quantization_mode="symmetric",
                smooth_quant_alpha=-1,  # Not sure what it does, but it is present in Stable Diffusion V2 example
            ),
        )
    logger.debug("Model quantization using NNCF completed")
    return quantized_model


def openvino_ptq(
    model,
    calibration_loader,
    quantization_skip_layers,
    calibration_batches: int = 16,
    validation_loader: Optional[DataLoader] = None,
    validation_fn: Optional[None] = None,
):
    import nncf
    import openvino as ov
    from super_gradients.training.utils.utils import infer_model_device

    device = infer_model_device(model)

    def transform_fn_to_device(data_item, device):
        images = data_item[0]
        return images.to(device)

    def transform_fn_to_numpy(data_item):
        images = data_item[0]
        return images.numpy()

    ignored_scope = None

    if quantization_skip_layers is not None:
        ignored_scope = nncf.IgnoredScope(patterns=list(quantization_skip_layers))
        logger.debug(f"Quantization skip layers: {quantization_skip_layers}")

    if validation_loader is not None and validation_fn is not None:
        logger.debug("Starting model quantization using NNCF with QC")

        calibration_dataset = nncf.Dataset(calibration_loader, transform_func=transform_fn_to_numpy)
        validation_dataset = nncf.Dataset(validation_loader, transform_func=transform_fn_to_numpy)

        example_input = next(iter(calibration_dataset.get_inference_data([0])))

        model = ov.convert_model(model, example_input=example_input)

        quantized_model = nncf.quantize_with_accuracy_control(
            model,
            calibration_dataset=calibration_dataset,
            validation_dataset=validation_dataset,
            validation_fn=validation_fn,
            ignored_scope=ignored_scope,
            subset_size=calibration_batches,
        )
        logger.debug("Model quantization using NNCF with QC completed")
    else:
        logger.debug("Starting model quantization using NNCF without QC")
        calibration_dataset = nncf.Dataset(calibration_loader, transform_func=partial(transform_fn_to_device, device=device))
        quantized_model = nncf.quantize(
            model,
            calibration_dataset=calibration_dataset,
            ignored_scope=ignored_scope,
            subset_size=calibration_batches,  # TODO: Check whether subset_size is sample size or batch size
            preset=nncf.QuantizationPreset.MIXED,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                # quantization_mode="symmetric",
                smooth_quant_alpha=-1,  # Not sure what it does but it is present in Stable Diffusion V2 example
            ),
        )
    logger.debug("Model quantization using NNCF completed")
    return quantized_model


def ptq(
    model,
    selective_quantizer: Optional[SelectiveQuantizer],
    calibration_loader: Optional[DataLoader],
    calibration_method: str = "percentile",
    calibration_batches: int = 16,
    calibration_percentile: float = 99.99,
    calibration_verbose: bool = False,
):
    """
    Perform Post Training Quantization (PTQ) on the model.

    :param model: Input model to quantize. This function always returns a new model, the input model is not modified.
    :param selective_quantizer:  An instance of SelectiveQuantizer class that defines what modules to quantize.
    :param calibration_loader: An instance of DataLoader that provides calibration data (optional).
    :param calibration_method: (str) Calibration method for quantized models. See QuantizationCalibrator for details.
    :param calibration_batches: (int) Number of batches to use for calibration. Default is 16.
    :param calibration_percentile: (float) Percentile for percentile calibration method. Default is 99.99.
    :param calibration_verbose:
    :return: A quantized model
    """
    contains_quantized_modules = check_model_contains_quantized_modules(model)
    if contains_quantized_modules:
        logger.debug("Model contains quantized modules. Skipping quantization & calibration steps since it is already quantized.")
        return model

    model = copy.deepcopy(model).eval()

    if selective_quantizer is None:
        selective_quantizer = SelectiveQuantizer(
            default_quant_modules_calibrator_weights="max",
            default_quant_modules_calibrator_inputs="histogram",
            default_per_channel_quant_weights=True,
            default_learn_amax=False,
            verbose=True,
        )
    selective_quantizer.quantize_module(model)

    if calibration_loader:
        logger.debug("Calibrating model")
        calibrator = QuantizationCalibrator(verbose=calibration_verbose)
        calibrator.calibrate_model(
            model,
            method=calibration_method,
            calib_data_loader=calibration_loader,
            num_calib_batches=calibration_batches,
            percentile=calibration_percentile,
        )
        logger.debug("Calibrating model complete")
        calibrator.reset_calibrators(model)

    return model
