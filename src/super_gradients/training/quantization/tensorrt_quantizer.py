import copy
import dataclasses
from typing import Union, List, Mapping, Optional

import torchmetrics
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_quantizer
from super_gradients.import_utils import import_pytorch_quantization_or_install
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches
from super_gradients.training.dataloaders import dataloaders
from super_gradients.training.quantization.abstract_quantizer import AbstractQuantizer, QuantizationResult
from super_gradients.training.utils.utils import get_param
from torch import nn
from torch.utils.data import DataLoader

logger = get_logger(__name__)

__all__ = ["TRTSelectiveQuantizationParams", "TRTQATParams", "TRTQuantizerCalibrationParams", "TRTQATQuantizer", "TRTPTQQuantizer"]


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
class TRTPTQQuantizer(AbstractQuantizer):
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
    >>>       histogram_calib_method: "percentile"  # calibration method for all "histogram" calibrators,
    >>>                                             # acceptable types are ["percentile", "entropy", "mse"], "max" calibrators always use "max"
    >>>       percentile: 99.99                     # percentile for all histogram calibrators with method "percentile", other calibrators are not affected
    >>>       num_calib_batches: 16                 # number of batches to use for calibration, if None, 512 / batch_size will be used
    >>>       verbose: False                        # if calibrator should be verbose
    """

    def __init__(
        self,
        selective_quantizer_params: Union[TRTSelectiveQuantizationParams, Mapping],
        calib_params: Union[TRTQuantizerCalibrationParams, Mapping],
    ):
        import_pytorch_quantization_or_install()

        if isinstance(selective_quantizer_params, Mapping):
            selective_quantizer_params = TRTSelectiveQuantizationParams(**selective_quantizer_params)
        if isinstance(calib_params, Mapping):
            calib_params = TRTQuantizerCalibrationParams(**calib_params)

        self.calib_params = calib_params
        self.selective_quantizer_params = selective_quantizer_params

    def quantize_from_config(
        self,
        cfg,
        model,
        trainer,
    ) -> QuantizationResult:
        from super_gradients.training.dataloaders import dataloaders

        val_dataloader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=copy.deepcopy(cfg.dataset_params.val_dataset_params),
            dataloader_params=copy.deepcopy(cfg.dataset_params.val_dataloader_params),
        )

        if "calib_dataloader" in cfg:
            calib_dataloader_name = get_param(cfg, "calib_dataloader")
            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.calib_dataloader_params)
            calib_dataset_params = copy.deepcopy(cfg.dataset_params.calib_dataset_params)
        else:
            calib_dataloader_name = get_param(cfg, "train_dataloader")

            calib_dataset_params = copy.deepcopy(cfg.dataset_params.train_dataset_params)
            calib_dataset_params.transforms = cfg.dataset_params.val_dataset_params.transforms

            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.train_dataloader_params)
            calib_dataloader_params.shuffle = False
            calib_dataloader_params.drop_last = False

        calib_dataloader = dataloaders.get(
            name=calib_dataloader_name,
            dataset_params=calib_dataset_params,
            dataloader_params=calib_dataloader_params,
        )

        return self.ptq(
            model=model,
            trainer=trainer,
            validation_loader=val_dataloader,
            validation_metrics=cfg.training_hyperparams.valid_metrics_list,
            calibration_loader=calib_dataloader,
        )

    def quantize_explicit(
        self,
        model: nn.Module,
        trainer,
        training_hyperparams: Optional[Mapping],
        train_loader: Optional[DataLoader],
        validation_loader: DataLoader,
        validation_metrics: List[torchmetrics.Metric],
        calibration_loader: DataLoader,
    ):
        return self.ptq(
            model=model,
            trainer=trainer,
            calibration_loader=calibration_loader,
            validation_loader=validation_loader,
            validation_metrics=validation_metrics,
        )

    def ptq(
        self,
        model: nn.Module,
        trainer,
        calibration_loader: DataLoader,
        validation_loader: DataLoader,
        validation_metrics,
    ):
        from super_gradients.training.utils.quantization.tensorrt.functional import tensorrt_ptq
        from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

        original_model = model
        model = copy.deepcopy(model).eval()
        fuse_repvgg_blocks_residual_branches(model)

        original_metrics = trainer.test(model=model, test_loader=validation_loader, test_metrics_list=validation_metrics)
        results = ["Original Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in original_metrics.items()]
        logger.info("\n".join(results))

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
        quantized_metrics = trainer.test(model=quantized_model, test_loader=validation_loader, test_metrics_list=validation_metrics)
        results = ["PTQ Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in quantized_metrics.items()]
        logger.info("\n".join(results))

        return QuantizationResult(
            original_model=original_model,
            original_metrics=original_metrics,
            quantized_model=quantized_model,
            quantized_metrics=quantized_metrics,
            calibration_dataloader=calibration_loader,
            export_path=None,
            export_result=None,
        )


@register_quantizer()
class TRTQATQuantizer(TRTPTQQuantizer):
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
    >>>       histogram_calib_method: "percentile"  # calibration method for all "histogram" calibrators,
    >>>                                             # acceptable types are ["percentile", "entropy", "mse"], "max" calibrators always use "max"
    >>>       percentile: 99.99                     # percentile for all histogram calibrators with method "percentile", other calibrators are not affected
    >>>       num_calib_batches: 16                 # number of batches to use for calibration, if None, 512 / batch_size will be used
    >>>       verbose: False                        # if calibrator should be verbose
    """

    def __init__(
        self,
        selective_quantizer_params: Union[TRTSelectiveQuantizationParams, Mapping],
        calib_params: Union[TRTQuantizerCalibrationParams, Mapping],
        qat_params: Union[TRTQATParams, Mapping, None] = None,
    ):
        import_pytorch_quantization_or_install()

        super().__init__(selective_quantizer_params=selective_quantizer_params, calib_params=calib_params)
        if isinstance(qat_params, Mapping):
            qat_params = TRTQATParams(**qat_params)
        self.qat_params = qat_params

    def quantize_from_config(
        self,
        *,
        cfg,
        model,
        trainer,
    ):
        num_gpus = get_param(cfg, "num_gpus")
        device = get_param(cfg, "device")
        if num_gpus != 1 and device == "cuda":
            raise NotImplementedError(
                f"Recipe requests multi_gpu={cfg.multi_gpu} and num_gpus={cfg.num_gpus}. QAT is proven to work correctly only with multi_gpu=OFF and num_gpus=1"
            )

        if self.qat_params is None:
            logger.warning(
                "QAT parameters are not provided. Using default QAT heuristics. "
                f"For optimal performance you may want to pass qat_params to {self.__class__.__name__}."
            )
            qat_params = TRTQATParams()
        else:
            qat_params = self.qat_params

        from super_gradients.training.utils.quantization.tensorrt.functional import modify_params_for_qat

        (training_hyperparams, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params,) = modify_params_for_qat(
            training_hyperparams=cfg.training_hyperparams,
            train_dataset_params=cfg.dataset_params.train_dataset_params,
            train_dataloader_params=cfg.dataset_params.train_dataloader_params,
            val_dataset_params=cfg.dataset_params.val_dataset_params,
            val_dataloader_params=cfg.dataset_params.val_dataloader_params,
            batch_size_divisor=qat_params.batch_size_divisor,
            disable_phase_callbacks=qat_params.disable_phase_callbacks,
            cosine_final_lr_ratio=qat_params.cosine_final_lr_ratio,
            warmup_epochs_divisor=qat_params.warmup_epochs_divisor,
            lr_decay_factor=qat_params.lr_decay_factor,
            max_epochs_divisor=qat_params.max_epochs_divisor,
            disable_augmentations=qat_params.disable_augmentations,
        )

        train_dataloader = dataloaders.get(
            name=get_param(cfg, "train_dataloader"),
            dataset_params=copy.deepcopy(train_dataset_params),
            dataloader_params=copy.deepcopy(train_dataloader_params),
        )

        validation_loader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=copy.deepcopy(val_dataset_params),
            dataloader_params=copy.deepcopy(val_dataloader_params),
        )

        if "calib_dataloader" in cfg:
            calib_dataloader_name = get_param(cfg, "calib_dataloader")
            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.calib_dataloader_params)
            calib_dataset_params = copy.deepcopy(cfg.dataset_params.calib_dataset_params)
        else:
            calib_dataloader_name = get_param(cfg, "train_dataloader")

            calib_dataset_params = copy.deepcopy(train_dataset_params)
            calib_dataset_params.transforms = val_dataset_params.transforms

            calib_dataloader_params = copy.deepcopy(train_dataloader_params)
            calib_dataloader_params.shuffle = False
            calib_dataloader_params.drop_last = False

        calibration_loader = dataloaders.get(
            name=calib_dataloader_name,
            dataset_params=calib_dataset_params,
            dataloader_params=calib_dataloader_params,
        )
        validation_metrics = cfg.training_hyperparams.valid_metrics_list

        return self.quantize_explicit(
            model=model,
            trainer=trainer,
            training_hyperparams=training_hyperparams,
            train_loader=train_dataloader,
            validation_loader=validation_loader,
            validation_metrics=validation_metrics,
            calibration_loader=calibration_loader,
        )

    def quantize_explicit(
        self,
        model,
        trainer,
        training_hyperparams,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        validation_metrics,
        calibration_loader: DataLoader,
    ):
        original_metrics = trainer.test(model=model, test_loader=validation_loader, test_metrics_list=validation_metrics)

        ptq_result = self.ptq(
            model=model,
            trainer=trainer,
            calibration_loader=calibration_loader,
            validation_loader=validation_loader,
            validation_metrics=validation_metrics,
        )

        trainer.train(
            model=ptq_result.quantized_model,
            train_loader=train_loader,
            valid_loader=validation_loader,
            training_params=training_hyperparams,
        )

        quantized_metrics = trainer.test(model=ptq_result.quantized_model, test_loader=validation_loader, test_metrics_list=validation_metrics)
        return QuantizationResult(
            original_model=model,
            original_metrics=original_metrics,
            quantized_model=ptq_result.quantized_model,
            quantized_metrics=quantized_metrics,
            calibration_dataloader=calibration_loader,
            export_result=None,
            export_path=None,
        )
