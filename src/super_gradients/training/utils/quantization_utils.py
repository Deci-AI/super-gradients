"""
Quantization utilities

Methods are based on:
 https://github.com/NVIDIA/TensorRT/blob/51a4297753d3e12d0eed864be52400f429a6a94c/tools/pytorch-quantization/examples/torchvision/classification_flow.py#L385

(Licensed under the Apache License, Version 2.0)
"""
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import models
from super_gradients.training.utils.callbacks import Phase, PhaseCallback, PhaseContext
import os
from enum import Enum
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
from super_gradients.training.utils import get_param
from super_gradients.training.utils.distributed_training_utils import get_local_rank, \
    get_world_size
from torch.distributed import all_gather

logger = get_logger(__name__)

try:
    from pytorch_quantization import nn as quant_nn, quant_modules
    from pytorch_quantization import calib
    from pytorch_quantization.tensor_quant import QuantDescriptor

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


class QuantizationLevel(str, Enum):
    FP32 = 'FP32'
    FP16 = 'FP16'
    INT8 = 'INT8'
    HYBRID = 'Hybrid'

    @staticmethod
    def from_string(quantization_level: str) -> Enum:
        quantization_level = quantization_level.lower()
        if quantization_level == 'fp32':
            return QuantizationLevel.FP32
        elif quantization_level == 'fp16':
            return QuantizationLevel.FP16
        elif quantization_level == 'int8':
            return QuantizationLevel.INT8
        elif quantization_level == 'hybrid':
            return QuantizationLevel.HYBRID
        else:
            raise NotImplementedError(f'Quantization Level: "{quantization_level}" is not supported')


def export_qat_onnx(model: torch.nn.Module, onnx_filename: str, input_shape: tuple,
                    per_channel_quantization: bool = False):
    """
    Method for exporting onnx after QAT.

    :param model: torch.nn.Module, model to export
    :param onnx_filename: str, target path for the onnx file,
    :param input_shape: tuple, input shape (usually BCHW)
    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    else:
        model.eval()
        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion()
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        # Export ONNX for multiple batch sizes
        logger.info("Creating ONNX file: " + onnx_filename)
        dummy_input = torch.randn(input_shape, device='cuda')
        opset_version = 13 if per_channel_quantization else 12
        torch.onnx.export(model, dummy_input, onnx_filename, verbose=False, opset_version=opset_version,
                          enable_onnx_checker=False,
                          do_constant_folding=True)


def calibrate_model(model: torch.nn.Module, calib_data_loader: torch.utils.data.DataLoader, method: str = "percentile",
                    num_calib_batches: int = 2, percentile: float = 99.99):
    """
    Calibrates torch model with quantized modules.

    :param model:               torch.nn.Module, model to perfrom the calibration on.
    :param calib_data_loader:   torch.utils.data.DataLoader, data loader of the calibration dataset.
    :param method:              str, One of [percentile, mse, entropy, max]. Statistics method for amax computation of the quantized modules
                                (Default=percentile).
    :param num_calib_batches:   int, number of batches to collect the statistics from.
    :param percentile:          float, percentile value to use when Trainer,quant_modules_calib_method='percentile'. Discarded when other methods are used
                                (Default=99.99).

    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    elif method in ["percentile", "mse", "entropy", "max"]:
        with torch.no_grad():
            _collect_stats(model, calib_data_loader, num_batches=num_calib_batches)

            # FOR PERCENTILE WE MUST PASS PERCENTILE VALUE THROUGH KWARGS,
            # SO IT WOULD BE PASSED TO module.load_calib_amax(**kwargs), AND IN OTHER METHODS WE MUST NOT PASS IT.
            if method == "precentile":
                _compute_amax(model, method="percentile", percentile=percentile)
            else:
                _compute_amax(model, method=method)
    else:
        raise ValueError(
            "Unsupported quantization calibration method, expected one of: percentile, mse, entropy, max, got " + str(
                method) + ".")


def _collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    else:
        local_rank = get_local_rank()
        world_size = get_world_size()

        # Enable calibrators
        _enable_calibrators(model)

        # Feed data to the network for collecting stats
        for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches, disable=local_rank > 0):
            if world_size > 1:
                all_batches = [torch.zeros_like(image, device='cuda') for _ in range(world_size)]
                all_gather(all_batches, image.cuda())
            else:
                all_batches = [image]

            for local_image in all_batches:
                model(local_image.cuda())
            if i >= num_batches:
                break

        # Disable calibrators
        _disable_calibrators(model)


def _disable_calibrators(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def _enable_calibrators(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()


def _compute_amax(model, **kwargs):
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    else:
        # Load calib result
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
        model.cuda()


def _deactivate_quant_modules_wrapping():
    """
    Deactivates quant modules wrapping, so that further modules won't use Q/DQ layers.
    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    else:
        quant_modules.deactivate()


def _activate_quant_modules_wrapping():
    """
    Activates quant modules wrapping, so that further modules use Q/DQ layers.
    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    else:
        quant_modules.initialize()


class QATCallback(PhaseCallback):
    """
    A callback for transitioning training into QAT.

    Rebuilds the model with QAT layers then either:
        1. loads the best checkpoint then performs calibration.
        2. loads an external calibrated model (makes sense when start_epoch=0).

    Additionally, resets Trainer's best_metric and sets ckpt_best_name to 'qat_ckpt_best.pth' so best QAT checkpoints
     will be saved separately.

    If performing calibration- the calibrated model is evaluated, and the metric_to_watch is logged under
     calibrated_model_{metric_to_watch}. The calibrated checkpoint is saved under ckpt_calibrated_{calibration_method}.pth


    Attributes:
        start_epoch: int, first epoch to start QAT.

        quant_modules_calib_method: str, One of [percentile, mse, entropy, max]. Statistics method for amax
         computation of the quantized modules (default=percentile).

        per_channel_quant_modules: bool, whether quant modules should be per channel (default=False).

        calibrate: bool, whether to perfrom calibration (default=False).

        calibrated_model_path: str, path to a calibrated checkpoint (default=None).

        calib_data_loader: torch.utils.data.DataLoader, data loader of the calibration dataset. When None,
         context.train_loader will be used (default=None).

        num_calib_batches: int, number of batches to collect the statistics from.

        percentile: float, percentile value to use when Trainer,quant_modules_calib_method='percentile'.
         Discarded when other methods are used (Default=99.99).



    """

    def __init__(self, start_epoch: int, quant_modules_calib_method: str = "percentile",
                 per_channel_quant_modules: bool = False, calibrate: bool = True, calibrated_model_path: str = None,
                 calib_data_loader: DataLoader = None, num_calib_batches: int = 2, percentile: float = 99.99):
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

    def _validate_args(self, start_epoch: int, quant_modules_calib_method: str, calibrate, calibrated_model_path):
        if _imported_pytorch_quantization_failure:
            raise _imported_pytorch_quantization_failure
        if start_epoch < 0:
            raise ValueError("start_epoch must be positive.")
        if quant_modules_calib_method not in ["percentile", "mse", "entropy", "max"]:
            raise ValueError(
                "Unsupported quantization calibration method, expected one of: percentile, mse, entropy, max, got " + str(
                    self.quant_modules_calib_method) + ".")
        if not calibrate and calibrated_model_path is None:
            logger.warning(
                "calibrate=False and no calibrated_model_path is given. QAT will be on an uncalibrated model.")

    def __call__(self, context: PhaseContext):
        if context.epoch == self.start_epoch:
            # REMOVE REFERENCES TO NETWORK AND CLEAN GPU MEMORY BEFORE BUILDING THE NEW NET
            context.context_methods.set_net(None)
            context.net = None
            torch.cuda.empty_cache()

            # BUILD THE SAME MODEL BUT WITH FAKE QUANTIZED MODULES, AND LOAD BEST CHECKPOINT TO IT
            self._initialize_quant_modules()

            if self.calibrated_model_path is not None:
                checkpoint_path = self.calibrated_model_path
            elif self.start_epoch > 0:
                checkpoint_path = os.path.join(context.ckpt_dir, 'ckpt_best.pth')

            qat_net = models.get(context.architecture, arch_params=context.arch_params.to_dict(), checkpoint_path=checkpoint_path)

            _deactivate_quant_modules_wrapping()

            # UPDATE CONTEXT'S NET REFERENCE
            context.net = context.context_methods.get_net()

            if self.calibrate:
                self._calibrate_model(context)

            # RESET THE BEST METRIC VALUE SO WE SAVE CHECKPOINTS AFTER THE EXPECTED QAT ACCURACY DEGRADATION
            context.context_methods._reset_best_metric()

            # SET NEW FILENAME FOR THE BEST CHECKPOINT SO WE DON'T OVERRIDE THE PREVIOUS ONES
            context.context_methods.set_ckpt_best_name('qat_ckpt_best.pth')

            # FINALLY, SET THE QAT NET TO CONTINUE TRAINING
            context.context_methods.set_net(qat_net)

    def _calibrate_model(self, context: PhaseContext):
        """
        Performs model calibration (collecting stats and setting amax for the fake quantized moduls)

        :param context: PhaseContext, current phase context.
        """
        self.calib_data_loader = self.calib_data_loader or context.train_loader
        calibrate_model(model=context.net,
                        calib_data_loader=self.calib_data_loader,
                        method=self.quant_modules_calib_method,
                        num_calib_batches=self.num_calib_batches,
                        percentile=self.percentile)
        method_desc = self.quant_modules_calib_method + '_' + str(
            self.percentile) if self.quant_modules_calib_method == 'percentile' else self.quant_modules_calib_method

        if not context.ddp_silent_mode:
            logger.info("Performing additional validation on calibrated model...")

        calibrated_valid_results = context.context_methods.validate_epoch(epoch=self.start_epoch, silent_mode=True)
        calibrated_acc = calibrated_valid_results[context.metric_idx_in_results_tuple]

        if not context.ddp_silent_mode:
            logger.info("Calibrate model " + context.metric_to_watch + ": " + str(calibrated_acc))
            context.sg_logger.add_checkpoint(tag='ckpt_calibrated_' + method_desc + '.pth',
                                             state_dict={"net": context.net.state_dict(), "acc": calibrated_acc})
            context.sg_logger.add_scalar("Calibrated_Model_" + context.metric_to_watch,
                                         calibrated_acc,
                                         global_step=self.start_epoch)

    def _initialize_quant_modules(self):
        """
        Initialize quant modules wrapping.
        """

        if _imported_pytorch_quantization_failure is not None:
            raise _imported_pytorch_quantization_failure
        else:
            if self.quant_modules_calib_method in ["percentile", "mse", "entropy"]:
                calib_method_type = 'histogram'
            else:
                calib_method_type = 'max'

            if self.per_channel_quant_modules:
                quant_desc_input = QuantDescriptor(calib_method=calib_method_type)
                quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
                quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
            else:
                quant_desc_input = QuantDescriptor(calib_method=calib_method_type, axis=None)
                quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
                quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
                quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

                quant_desc_weight = QuantDescriptor(calib_method=calib_method_type, axis=None)
                quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
                quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
                quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

            _activate_quant_modules_wrapping()


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

            load_checkpoint_to_model(ckpt_local_path=best_ckpt_path,
                                     net=context.net,
                                     load_weights_only=True,
                                     load_ema_as_net=context.training_params.ema,
                                     strict=True,
                                     load_backbone=False
                                     )
            per_channel_quant_modules = get_param(context.training_params.qat_params, "per_channel_quant_modules")
            export_qat_onnx(context.net.module, onnx_path, self.dummy_input_size, per_channel_quant_modules)

            context.sg_logger.add_file("qat_ckpt_best.onnx")
