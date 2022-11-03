from torch.utils.data import DataLoader
import torch
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
from super_gradients.training.utils.callbacks import Phase, PhaseCallback, PhaseContext
import os
from enum import Enum
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model, read_ckpt_state_dict
from super_gradients.training.utils.quantization.core import QuantizedMetadata
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx

from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

logger = get_logger(__name__)

try:
    from pytorch_quantization import nn as quant_nn, quant_modules  # noqa: F401
    from pytorch_quantization import calib  # noqa: F401
    from pytorch_quantization.tensor_quant import QuantDescriptor  # noqa: F401

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


class QuantizationLevel(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"
    HYBRID = "Hybrid"

    @staticmethod
    def from_string(quantization_level: str) -> Enum:
        quantization_level = quantization_level.lower()
        if quantization_level == "fp32":
            return QuantizationLevel.FP32
        elif quantization_level == "fp16":
            return QuantizationLevel.FP16
        elif quantization_level == "int8":
            return QuantizationLevel.INT8
        elif quantization_level == "hybrid":
            return QuantizationLevel.HYBRID
        else:
            raise NotImplementedError(f'Quantization Level: "{quantization_level}" is not supported')


class QATCallback(PhaseCallback):
    """
    A callback for transitioning training into QAT.

    Rebuilds the model with QAT layers then either:
        1. loads the best checkpoint then performs calibration.
        2. loads an external calibrated model (makes sense when start_epoch=0).

    Additionally, resets SgModel's best_metric and sets ckpt_best_name to 'qat_ckpt_best.pth' so best QAT checkpoints
     will be saved separately.

    If performing calibration- the calibrated model is evaluated, and the metric_to_watch is logged under
     calibrated_model_{metric_to_watch}. The calibrated checkpoint is saved under ckpt_calibrated_{calibration_method}.pth


    Attributes:
        start_epoch: int, first epoch to start QAT.

        quant_modules_calib_method: str, One of [percentile, mse, entropy, max]. Statistics method for amax
         computation of the quantized modules (default=max).

        per_channel_quant_modules: bool, whether quant modules should be per channel (default=False).

        calibrate: bool, whether to perform calibration (default=True).

        calibrated_model_path: str, path to a calibrated checkpoint (default=None).

        calib_data_loader: torch.utils.data.DataLoader, data loader of the calibration dataset. When None,
         context.train_loader will be used (default=None).

        num_calib_batches: int, number of batches to collect the statistics from.

        percentile: float, percentile value to use when SgModel,quant_modules_calib_method='percentile'.
         Discarded when other methods are used (Default=99.99).



    """

    def __init__(
        self,
        start_epoch: int,
        quantization_mappings: list,
        quant_modules_calib_method: str = "max",
        per_channel_quant_modules: bool = True,
        calibrate: bool = True,
        calibrated_model_path: str = None,
        calib_data_loader: DataLoader = None,
        num_calib_batches: int = 2,
        percentile: float = 99.99,
    ):
        super(QATCallback, self).__init__(Phase.TRAIN_EPOCH_START)
        self._validate_args(start_epoch, quant_modules_calib_method, calibrate, calibrated_model_path)
        self.start_epoch = start_epoch
        self.quant_modules_calib_method = quant_modules_calib_method
        self.per_channel_quant_modules = per_channel_quant_modules
        self.calibrate = calibrate
        self.calibrated_model_path = calibrated_model_path
        self.calib_data_loader = calib_data_loader
        self.num_calib_batches = num_calib_batches
        self.percentile = percentile
        self.selective_quantizer = SelectiveQuantizer(
            custom_mappings=self._process_quantization_mappings(quantization_mappings),
            default_quant_modules_calib_method=quant_modules_calib_method,
            default_per_channel_quant_modules=per_channel_quant_modules,
        )
        self.calibrator = QuantizationCalibrator()

    def _process_quantization_mappings(self, quantization_mappings):
        mappings = dict()
        for mapping in quantization_mappings:
            if not isinstance(mapping, QuantizedMetadata):
                mapping = QuantizedMetadata.from_dict(mapping)
            mappings[mapping.float_source] = mapping

        return mappings

    def _validate_args(self, start_epoch: int, quant_modules_calib_method: str, calibrate, calibrated_model_path):
        if _imported_pytorch_quantization_failure:
            raise _imported_pytorch_quantization_failure
        if start_epoch < 0:
            raise ValueError("start_epoch must be positive.")

        accepted_methods = ["percentile", "mse", "entropy", "max"]

        if quant_modules_calib_method not in accepted_methods:
            raise ValueError(
                f"Unsupported quantization calibration method, "
                f"expected one of: {', '.join(accepted_methods)}, however, received: {quant_modules_calib_method}"
            )
        if not calibrate and calibrated_model_path is None:
            logger.warning("calibrate=False and no calibrated_model_path is given. QAT will be on an uncalibrated model.")

    def __call__(self, context: PhaseContext):
        if context.epoch == self.start_epoch:
            # SET CHECKPOINT PARAMS SO WE LOAD THE BEST CHECKPOINT SO FAR
            checkpoint_params_qat = context.checkpoint_params.to_dict()
            checkpoint_params_qat["ckpt_name"] = "ckpt_best.pth"

            if self.calibrated_model_path is not None:
                checkpoint_params_qat["external_checkpoint_path"] = self.calibrated_model_path
                checkpoint_params_qat["load_ema_as_net"] = "ema_net" in read_ckpt_state_dict(self.calibrated_model_path).keys()
                checkpoint_params_qat["load_checkpoint"] = True
            elif self.start_epoch > 0:
                checkpoint_params_qat["load_ema_as_net"] = context.training_params.ema
                checkpoint_params_qat["load_checkpoint"] = True
                if checkpoint_params_qat["load_ema_as_net"]:
                    logger.warning("EMA net loaded from best checkpoint, continuing QAT without EMA.")
                    context.context_methods.set_ema(False)

            # CLEAN GPU MEMORY BEFORE BUILDING THE NEW NET
            torch.cuda.empty_cache()

            # GET MODULE FOR QUANTIZATION
            module: torch.nn.Module = context.context_methods.get_net
            state_dict = module.state_dict()
            self.selective_quantizer.quantize_module(module)

            # FROM THIS MOMENT, `module` IS QUANTIZED - WE LOAD WEIGHTS
            module.load_state_dict(state_dict, strict=True)

            if self.calibrate:
                self._calibrate_model(context)

            # RESET THE BEST METRIC VALUE SO WE SAVE CHECKPOINTS AFTER THE EXPECTED QAT ACCURACY DEGRADATION
            context.context_methods.reset_best_metric()

            # SET NEW FILENAME FOR THE BEST CHECKPOINT SO WE DON'T OVERRIDE THE PREVIOUS ONES
            context.context_methods.set_ckpt_best_name("qat_ckpt_best.pth")

    def _calibrate_model(self, context: PhaseContext):
        """
        Performs model calibration (collecting stats and setting amax for the fake quantized moduls)

        :param context: PhaseContext, current phase context.
        """
        self.calib_data_loader = self.calib_data_loader or context.train_loader
        self.calibrator.calibrate_model(
            model=context.net,
            calib_data_loader=self.calib_data_loader,
            method=self.quant_modules_calib_method,
            num_calib_batches=self.num_calib_batches,
            percentile=self.percentile,
        )
        method_desc = (
            self.quant_modules_calib_method + "_" + str(self.percentile) if self.quant_modules_calib_method == "percentile" else self.quant_modules_calib_method
        )

        if not context.ddp_silent_mode:
            logger.info("Performing additional validation on calibrated model...")

        calibrated_valid_results = context.context_methods.validate_epoch(epoch=self.start_epoch, silent_mode=True)
        calibrated_acc = calibrated_valid_results[context.metric_idx_in_results_tuple]

        if not context.ddp_silent_mode:
            logger.info("Calibrate model " + context.metric_to_watch + ": " + str(calibrated_acc))
            context.sg_logger.add_checkpoint(tag="ckpt_calibrated_" + method_desc + ".pth", state_dict={"net": context.net.state_dict(), "acc": calibrated_acc})
            context.sg_logger.add_scalar("Calibrated_Model_" + context.metric_to_watch, calibrated_acc, global_step=self.start_epoch)


class PostQATConversionCallback(PhaseCallback):
    """
    Post QAT training callback that saves the best checkpoint (i.e ckpt_best.pth) in onnx format.
    Should be used with QATCallback.

    Attributes:
        dummy_input_size: (tuple) dummy input size for the ONNX conversion.
    """

    def __init__(self, dummy_input_size):
        super().__init__(phase=Phase.POST_TRAINING)
        self.dummy_input_size = dummy_input_size

    def __call__(self, context: PhaseContext):
        if not context.ddp_silent_mode:
            best_ckpt_path = os.path.join(context.ckpt_dir, "qat_ckpt_best.pth")
            onnx_path = os.path.join(context.ckpt_dir, "qat_ckpt_best.onnx")

            load_checkpoint_to_model(
                ckpt_local_path=best_ckpt_path,
                net=context.net,
                load_weights_only=True,
                load_ema_as_net=context.training_params.ema,
                strict=True,
                load_backbone=False,
            )
            export_quantized_module_to_onnx(context.net.module, onnx_path, self.dummy_input_size)

            context.sg_logger.add_file("qat_ckpt_best.onnx")
