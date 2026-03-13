import copy
from typing import Optional

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.quantization import SelectiveQuantizer, QuantizationCalibrator
from super_gradients.training.utils.utils import check_model_contains_quantized_modules, get_param
from torch import nn
from torch.utils.data import DataLoader

logger = get_logger(__name__)

__all__ = ["tensorrt_ptq", "modify_params_for_qat"]


def tensorrt_ptq(
    model: nn.Module,
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
           Method will always switch model to eval mode. Fusion of repvgg blocks should be done before calling this method if needed.
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
