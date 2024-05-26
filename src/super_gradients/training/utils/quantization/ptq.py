import copy
from typing import Optional

from torch.utils.data import DataLoader

from .calibrator import QuantizationCalibrator
from .selective_quantization_utils import SelectiveQuantizer
from super_gradients.training.utils.utils import check_model_contains_quantized_modules
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


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
