import copy
import dataclasses
import os
from typing import Union, List, Mapping, Dict, Optional

import torch.cuda
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path
from super_gradients.common.registry.registry import register_quantizer
from super_gradients.import_utils import import_pytorch_quantization_or_install
from super_gradients.training import modify_params_for_qat
from super_gradients.training.utils.utils import check_model_contains_quantized_modules, get_param
from torch import nn
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches
from super_gradients.training.utils.quantization import SelectiveQuantizer, QuantizationCalibrator
from super_gradients.training.utils.quantization.abstract_quantizer import AbstractQuantizer, QuantizationResult
from torch.utils.data import DataLoader
from torchmetrics import Metric

logger = get_logger(__name__)


@dataclasses.dataclass
class TRTSelectiveQuantizationParams:
    """
    calibrator_w: "max"        # calibrator type for weights, acceptable types are ["max", "histogram"]
    calibrator_i: "histogram"  # calibrator type for inputs acceptable types are ["max", "histogram"]
    per_channel: True          # per-channel quantization of weights, activations stay per-tensor by default
    learn_amax: False          # enable learnable amax in all TensorQuantizers using straight-through estimator
    """

    calibrator_w: str = "max"
    calibrator_i: str = "histogram"
    per_channel: bool = True
    learn_amax: bool = False
    skip_modules: Union[List[str], None] = None


@dataclasses.dataclass
class TRTQuantizerCalibrationParams:
    """
    :param histogram_calib_method: calibration method for all "histogram" calibrators,
    acceptable types are ["percentile", "entropy", "mse"], "max" calibrators always use "max"
    :param percentile: percentile for all histogram calibrators with method "percentile", other calibrators are not affected
    :param num_calib_batches: number of batches to use for calibration, if None, 512 / batch_size will be used
    :param verbose: if calibrator should be verbose
    """

    histogram_calib_method: str = "percentile"
    percentile: float = 99.9
    num_calib_batches: Union[int, None] = 128
    verbose: bool = False


@dataclasses.dataclass
class TRTQATParams:
    """
    :param int batch_size_divisor: Divisor used to calculate the batch size. Default value is 2.
    :param int max_epochs_divisor: Divisor used to calculate the maximum number of epochs. Default value is 10.
    :param float lr_decay_factor: Factor used to decay the learning rate, weight decay and warmup. Default value is 0.01.
    :param int warmup_epochs_divisor: Divisor used to calculate the number of warm-up epochs. Default value is 10.
    :param float cosine_final_lr_ratio: Ratio used to determine the final learning rate in a cosine annealing schedule. Default value is 0.01.
    :param bool disable_phase_callbacks: Flag to control to disable phase callbacks, which can interfere with QAT. Default value is True.
    :param bool disable_augmentations: Flag to control to disable phase augmentations, which can interfere with QAT. Default value is False.
    """

    batch_size_divisor: int = 2
    max_epochs_divisor: int = 10
    lr_decay_factor: float = 0.01
    warmup_epochs_divisor: int = 10
    cosine_final_lr_ratio: float = 0.01
    disable_phase_callbacks: bool = True
    disable_augmentations: bool = False


@register_quantizer()
class TRTQuantizer(AbstractQuantizer):
    """
    >>> trt_quantizer_params.yaml
    >>> ptq_only: False              # whether to launch QAT, or leave PTQ only
    >>> quantizer:
    >>>   TRTQuantizer:
    >>>     selective_quantizer_params:
    >>>       calibrator_w: "max"        # calibrator type for weights, acceptable types are ["max", "histogram"]
    >>>       calibrator_i: "histogram"  # calibrator type for inputs acceptable types are ["max", "histogram"]
    >>>       per_channel: True          # per-channel quantization of weights, activations stay per-tensor by default
    >>>       learn_amax: False          # enable learnable amax in all TensorQuantizers using straight-through estimator
    >>>       skip_modules:              # optional list of module names (strings) to skip from quantization
    >>>
    >>>     calib_params:
    >>>       histogram_calib_method: "percentile"  # calibration method for all "histogram" calibrators, acceptable types are ["percentile", "entropy", "mse"], "max" calibrators always use "max"
    >>>       percentile: 99.99                     # percentile for all histogram calibrators with method "percentile", other calibrators are not affected
    >>>       num_calib_batches: 16                 # number of batches to use for calibration, if None, 512 / batch_size will be used
    >>>       verbose: False                        # if calibrator should be verbose
    """

    def __init__(
        self,
        selective_quantizer_params: Union[TRTSelectiveQuantizationParams, Mapping],
        calib_params: Union[TRTQuantizerCalibrationParams, Mapping],
        qat_params: Union[TRTQATParams, Mapping],
    ):
        import_pytorch_quantization_or_install()

        if isinstance(selective_quantizer_params, Mapping):
            selective_quantizer_params = TRTSelectiveQuantizationParams(**selective_quantizer_params)
        if isinstance(calib_params, Mapping):
            calib_params = TRTQuantizerCalibrationParams(**calib_params)
        if isinstance(qat_params, Mapping):
            qat_params = TRTQATParams(**qat_params)

        self.calib_params = calib_params
        self.selective_quantizer_params = selective_quantizer_params
        self.qat_params = qat_params

    def ptq(
        self,
        model: nn.Module,
        trainer: "Trainer",
        calibration_loader: DataLoader,
        validation_loader: DataLoader,
        validation_metrics,
    ):
        original_model = model
        model = copy.deepcopy(model).eval()
        fuse_repvgg_blocks_residual_branches(model)

        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights=self.selective_quantizer_params.calibrator_w,
            default_quant_modules_calibrator_inputs=self.selective_quantizer_params.calibrator_i,
            default_per_channel_quant_weights=self.selective_quantizer_params.per_channel,
            default_learn_amax=self.selective_quantizer_params.learn_amax,
            verbose=self.calib_params.verbose,
        )
        q_util.register_skip_quantization(layer_names=self.selective_quantizer_params.skip_modules)

        quantized_model = tensorrt_ptq(
            model,
            selective_quantizer=q_util,
            calibration_loader=calibration_loader,
            calibration_method=self.calib_params.histogram_calib_method,
            calibration_batches=self.calib_params.num_calib_batches or max(1, int(512 // calibration_loader.batch_size)),
            calibration_percentile=self.calib_params.percentile,
            calibration_verbose=self.calib_params.verbose,
        )

        # VALIDATE PTQ MODEL AND PRINT SUMMARY
        logger.info("Validating PTQ model...")
        metrics = trainer.test(model=quantized_model, test_loader=validation_loader, test_metrics_list=validation_metrics)
        results = ["PTQ Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in metrics.items()]
        logger.info("\n".join(results))

        return QuantizationResult(original_model=original_model, quantized_model=quantized_model, metrics=metrics)

    def qat(
        self,
        *,
        model: nn.Module,
        trainer: "Trainer",
        train_loader: DataLoader,
        training_hyperparams,
        validation_loader: DataLoader,
        calibration_loader: DataLoader = None,
        training_params: Mapping,
        additional_qat_configs_to_log: Dict = None,
        validation_metrics: List[Metric],
    ):
        (training_hyperparams, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params,) = modify_params_for_qat(
            training_hyperparams=training_hyperparams,
            train_dataset_params=cfg.dataset_params.train_dataset_params,
            train_dataloader_params=cfg.dataset_params.train_dataloader_params,
            val_dataset_params=cfg.dataset_params.val_dataset_params,
            val_dataloader_params=cfg.dataset_params.val_dataloader_params,
            quantization_params=cfg.quantization_params,
            batch_size_divisor=self.qat_params.batch_size_divisor,
            disable_phase_callbacks=self.qat_params.disable_phase_callbacks,
            cosine_final_lr_ratio=self.qat_params.cosine_final_lr_ratio,
            warmup_epochs_divisor=self.qat_params.warmup_epochs_divisor,
            lr_decay_factor=self.qat_params.lr_decay_factor,
            max_epochs_divisor=self.qat_params.max_epochs_divisor,
            disable_augmentations=self.qat_params.disable_augmentations,
        )

        ptq_result = self.ptq(
            model=model,
            trainer=trainer,
            calibration_loader=calibration_loader,
            validation_loader=validation_loader,
            validation_metrics=validation_metrics,
        )
        # TRAIN
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        run_id = get_param(trainer.training_params, "run_id", None)
        logger.debug(f"Experiment run id {run_id}")
        output_dir_path = get_checkpoints_dir_path(ckpt_root_dir=trainer.ckpt_root_dir, experiment_name=trainer.experiment_name, run_id=run_id)
        logger.debug(f"Output directory {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)

        trainer.train(
            model=ptq_result.quantized_model,
            train_loader=train_loader,
            valid_loader=validation_loader,
            training_params=training_params,
            additional_configs_to_log=additional_qat_configs_to_log,
        )

        metrics = trainer.test(model=model, test_loader=validation_loader, test_metrics_list=validation_metrics)
        return QuantizationResult(original_model=model, quantized_model=model, metrics=metrics)


def tensorrt_ptq(
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
