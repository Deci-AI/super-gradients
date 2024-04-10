import copy
import dataclasses
import os
from typing import Union, List, Mapping, Optional

import torch.cuda
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path
from super_gradients.common.registry.registry import register_quantizer
from super_gradients.import_utils import import_pytorch_quantization_or_install
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches
from super_gradients.training.dataloaders import dataloaders
from super_gradients.training.utils.quantization import SelectiveQuantizer, QuantizationCalibrator
from super_gradients.training.utils.quantization.abstract_quantizer import AbstractQuantizer, QuantizationResult
from super_gradients.training.utils.utils import check_model_contains_quantized_modules, get_param
from torch import nn
from torch.utils.data import DataLoader

logger = get_logger(__name__)

__all__ = ["TRTSelectiveQuantizationParams", "TRTQATParams", "TRTQuantizerCalibrationParams", "TRTQuantizer"]


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
        trainer,
        calibration_loader: DataLoader,
        validation_loader: DataLoader,
        validation_metrics,
    ):
        original_model = model
        model = copy.deepcopy(model).eval()
        fuse_repvgg_blocks_residual_branches(model)

        original_metrics = trainer.test(model=model, test_loader=validation_loader, test_metrics_list=validation_metrics)

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
            quantized_model=quantized_model,
            quantized_metrics=quantized_metrics,
            original_metrics=original_metrics,
            exported_model_path=None,
            export_result=None,
        )

    def qat(
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

        (training_hyperparams, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params,) = modify_params_for_qat(
            training_hyperparams=cfg.training_hyperparams,
            train_dataset_params=cfg.dataset_params.train_dataset_params,
            train_dataloader_params=cfg.dataset_params.train_dataloader_params,
            val_dataset_params=cfg.dataset_params.val_dataset_params,
            val_dataloader_params=cfg.dataset_params.val_dataloader_params,
            batch_size_divisor=self.qat_params.batch_size_divisor,
            disable_phase_callbacks=self.qat_params.disable_phase_callbacks,
            cosine_final_lr_ratio=self.qat_params.cosine_final_lr_ratio,
            warmup_epochs_divisor=self.qat_params.warmup_epochs_divisor,
            lr_decay_factor=self.qat_params.lr_decay_factor,
            max_epochs_divisor=self.qat_params.max_epochs_divisor,
            disable_augmentations=self.qat_params.disable_augmentations,
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
            train_loader=train_dataloader,
            valid_loader=validation_loader,
            training_params=training_hyperparams,
            additional_configs_to_log=cfg,
        )

        metrics = trainer.test(model=model, test_loader=validation_loader, test_metrics_list=validation_metrics)
        return QuantizationResult(original_model=model, quantized_model=model, metrics=metrics)

    def export(self, *, original_model, quantization_result, exporter):
        # TODO: Add export functionality
        return quantization_result


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


def modify_params_for_qat(
    training_hyperparams,
    train_dataset_params,
    val_dataset_params,
    train_dataloader_params,
    val_dataloader_params,
    batch_size_divisor: int = 2,
    max_epochs_divisor: int = 10,
    lr_decay_factor: float = 0.01,
    warmup_epochs_divisor: int = 10,
    cosine_final_lr_ratio: float = 0.01,
    disable_phase_callbacks: bool = True,
    disable_augmentations: bool = False,
):
    """

    This method modifies the recipe for QAT to implement rules of thumb based on the regular non-qat recipe.
    It does so by manipulating the training_hyperparams, train_dataloader_params, val_dataloader_params, train_dataset_params, val_dataset_params.
    Usage:
        trainer = Trainer("test_launch_qat_with_minimal_changes")
        net = ResNet18(num_classes=10, arch_params={})
        train_params = {...}

        train_dataset_params = {
            "transforms": [...
            ]
        }

        train_dataloader_params = {"batch_size": 256}

        val_dataset_params = {"transforms": [ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]}

        val_dataloader_params = {"batch_size": 256}

        train_loader = cifar10_train(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        valid_loader = cifar10_val(dataset_params=val_dataset_params, dataloader_params=val_dataloader_params)

        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params = modify_params_for_qat(
            train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params
        )

        train_loader = cifar10_train(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        valid_loader = cifar10_val(dataset_params=val_dataset_params, dataloader_params=val_dataloader_params)

        trainer.qat(
            model=net,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
            calib_loader=train_loader,
        )

    :param val_dataset_params: Dict, validation dataset_params to be passed to dataloaders.get(...) when instantiating the train dataloader.
    :param train_dataset_params: Dict, train dataset_params to be passed to dataloaders.get(...) when instantiating the validation dataloader.
    :param val_dataloader_params: Dict, validation dataloader_params to be passed to dataloaders.get(...) when instantiating the validation dataloader.
    :param train_dataloader_params: Dict, train dataloader_params to be passed to dataloaders.get(...) when instantiating the train dataloader.
    :param training_hyperparams: Dict, train parameters passed to Trainer.qat(...)
    :param int batch_size_divisor: Divisor used to calculate the batch size. Default value is 2.
    :param int max_epochs_divisor: Divisor used to calculate the maximum number of epochs. Default value is 10.
    :param float lr_decay_factor: Factor used to decay the learning rate, weight decay and warmup. Default value is 0.01.
    :param int warmup_epochs_divisor: Divisor used to calculate the number of warm-up epochs. Default value is 10.
    :param float cosine_final_lr_ratio: Ratio used to determine the final learning rate in a cosine annealing schedule. Default value is 0.01.
    :param bool disable_phase_callbacks: Flag to control to disable phase callbacks, which can interfere with QAT. Default value is True.
    :param bool disable_augmentations: Flag to control to disable phase augmentations, which can interfere with QAT. Default value is False.
    :return: modified (copy) training_hyperparams, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params
    """

    training_hyperparams = copy.deepcopy(training_hyperparams)
    train_dataloader_params = copy.deepcopy(train_dataloader_params)
    val_dataloader_params = copy.deepcopy(val_dataloader_params)
    train_dataset_params = copy.deepcopy(train_dataset_params)
    val_dataset_params = copy.deepcopy(val_dataset_params)

    if "max_epochs" not in training_hyperparams.keys():
        raise ValueError("max_epochs is a required field in training_hyperparams for QAT modification.")

    if "initial_lr" not in training_hyperparams.keys():
        raise ValueError("initial_lr is a required field in training_hyperparams for QAT modification.")

    if "optimizer_params" not in training_hyperparams.keys():
        raise ValueError("optimizer_params is a required field in training_hyperparams for QAT modification.")

    if "weight_decay" not in training_hyperparams["optimizer_params"].keys():
        raise ValueError("weight_decay is a required field in training_hyperparams['optimizer_params'] for QAT modification.")

    # Q/DQ Layers take a lot of space for activations in training mode
    train_dataloader_params["batch_size"] = max(1, train_dataloader_params["batch_size"] // batch_size_divisor)
    val_dataloader_params["batch_size"] = max(1, val_dataloader_params["batch_size"] // batch_size_divisor)

    logger.info(f"New dataset_params.train_dataloader_params.batch_size: {train_dataloader_params['batch_size']}")
    logger.info(f"New dataset_params.val_dataloader_params.batch_size: {val_dataloader_params['batch_size']}")

    training_hyperparams["max_epochs"] = max(1, training_hyperparams["max_epochs"] // max_epochs_divisor)
    logger.warning(f"New number of epochs: {training_hyperparams['max_epochs']}")
    training_hyperparams["initial_lr"] *= lr_decay_factor
    if get_param(training_hyperparams, "warmup_initial_lr") is not None:
        training_hyperparams["warmup_initial_lr"] *= lr_decay_factor
    else:
        training_hyperparams["warmup_initial_lr"] = training_hyperparams["initial_lr"] * 0.01
    training_hyperparams["optimizer_params"]["weight_decay"] *= lr_decay_factor
    logger.warning(f"New learning rate: {training_hyperparams['initial_lr']}")
    logger.warning(f"New weight decay: {training_hyperparams['optimizer_params']['weight_decay']}")
    # as recommended by pytorch-quantization docs
    if get_param(training_hyperparams, "lr_mode") != "CosineLRScheduler":
        training_hyperparams["lr_mode"] = "CosineLRScheduler"
    training_hyperparams["cosine_final_lr_ratio"] = cosine_final_lr_ratio
    logger.warning(
        f"lr_mode will be set to cosine for QAT run instead of {get_param(training_hyperparams, 'lr_mode')} with "
        f"cosine_final_lr_ratio={cosine_final_lr_ratio}"
    )

    training_hyperparams["lr_warmup_epochs"] = (training_hyperparams["max_epochs"] // warmup_epochs_divisor) or 1
    logger.warning(f"New lr_warmup_epochs: {training_hyperparams['lr_warmup_epochs']}")

    # do mess with Q/DQ
    if get_param(training_hyperparams, "average_best_models"):
        logger.info("Model averaging will be disabled for QAT run.")
        training_hyperparams["average_best_models"] = False
    if get_param(training_hyperparams, "ema"):
        logger.warning("EMA will be disabled for QAT run.")
        training_hyperparams["ema"] = False
    if get_param(training_hyperparams, "sync_bn"):
        logger.warning("SyncBatchNorm will be disabled for QAT run.")
        training_hyperparams["sync_bn"] = False
    if disable_phase_callbacks and get_param(training_hyperparams, "phase_callbacks") is not None and len(training_hyperparams["phase_callbacks"]) > 0:
        logger.warning(f"Recipe contains {len(training_hyperparams['phase_callbacks'])} phase callbacks. All of them will be disabled.")
        training_hyperparams["phase_callbacks"] = []
    # no augmentations
    if disable_augmentations and "transforms" in val_dataset_params:
        logger.warning("Augmentations will be disabled for QAT run. Using validation transforms instead.")
        train_dataset_params["transforms"] = val_dataset_params["transforms"]

    return training_hyperparams, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params
