import copy
import dataclasses
from functools import partial
from typing import Optional, Mapping
from typing import Union, List

import nncf
from nncf import TargetDevice, QuantizationPreset
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_quantizer
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches
from super_gradients.training.utils.quantization.abstract_quantizer import AbstractQuantizer, QuantizationResult
from super_gradients.training.utils.utils import infer_model_device, get_param
from torch import nn
from torch.utils.data import DataLoader

logger = get_logger(__name__)


@dataclasses.dataclass
class OpenVinoSelectiveQuantizationParams:
    """ """

    skip_patterns: Union[List[str], None] = None
    skip_types: Union[List[str], None] = None


@dataclasses.dataclass
class OpenVinoQuantizationParams:
    """ """

    target_device: Optional[str] = "ANY"
    preset: Optional[str] = None
    fast_bias_correction: bool = True


@dataclasses.dataclass
class OpenVinoCalibrationParams:
    """ """

    num_calib_batches: int = 128


@register_quantizer()
class OpenVinoQuantizer(AbstractQuantizer):
    def __init__(
        self,
        calib_params: OpenVinoCalibrationParams,
        quantizer_params: OpenVinoQuantizationParams,
        selective_quantizer_params: OpenVinoSelectiveQuantizationParams,
    ):
        super().__init__()
        if isinstance(calib_params, Mapping):
            calib_params = OpenVinoCalibrationParams(**calib_params)
        if isinstance(selective_quantizer_params, Mapping):
            selective_quantizer_params = OpenVinoSelectiveQuantizationParams(**selective_quantizer_params)
        if isinstance(quantizer_params, Mapping):
            quantizer_params = OpenVinoQuantizationParams(**quantizer_params)

        self.selective_quantization_params = selective_quantizer_params
        self.calibration_params = calib_params
        self.quantizer_params = quantizer_params

    def quantize(
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

        quantized_model = openvino_ptq(
            model=model,
            calibration_loader=calibration_loader,
            calibration_batches=self.calibration_params.num_calib_batches,
            skip_patterns=self.selective_quantization_params.skip_patterns,
            skip_types=self.selective_quantization_params.skip_types,
            target_device=self.quantizer_params.target_device,
            preset=self.quantizer_params.preset,
            fast_bias_correction=self.quantizer_params.fast_bias_correction,
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

    def qat(self, *args, **kwargs):
        raise NotImplementedError("QAT is not supported for OpenVinoQuantizer")


def openvino_ptq(
    model: nn.Module,
    calibration_loader,
    skip_patterns: Optional[List[str]],
    skip_types: Optional[List[str]],
    calibration_batches: int,
    preset,
    target_device,
    fast_bias_correction: bool,
):
    device = infer_model_device(model)

    def transform_fn_to_device(data_item, device):
        images = data_item[0]
        return images.to(device)

    ignored_scope_dict = dict(
        patterns=list(skip_patterns) if skip_patterns is not None else None,
        types=list(skip_types) if skip_types is not None else None,
    )
    ignored_scope = nncf.IgnoredScope(**ignored_scope_dict)
    if skip_patterns is not None:
        logger.debug(f"Quantization skip layers: {skip_patterns}")
    if skip_types is not None:
        logger.debug(f"Quantization skip types: {skip_types}")

    logger.debug("Starting model quantization using NNCF without QC")
    calibration_dataset = nncf.Dataset(calibration_loader, transform_func=partial(transform_fn_to_device, device=device))
    calibration_loader_len = len(calibration_loader)
    if calibration_batches > calibration_loader_len:
        logger.warning(
            f"Calibration batches ({calibration_batches}) is greater than the number of batches in the calibration loader ({calibration_loader_len})."
        )
        calibration_batches = calibration_loader_len

    quantized_model = nncf.quantize(
        model,
        calibration_dataset=calibration_dataset,
        ignored_scope=ignored_scope,
        subset_size=calibration_batches,
        preset=QuantizationPreset(preset) if preset is not None else None,
        target_device=TargetDevice(target_device) if target_device is not None else TargetDevice.ANY,
        fast_bias_correction=fast_bias_correction,
    )

    logger.debug("Model quantization using NNCF completed")
    return quantized_model
