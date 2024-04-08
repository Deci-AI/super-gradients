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
from super_gradients.training.utils.utils import infer_model_device
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
        metrics = trainer.test(model=quantized_model, test_loader=validation_loader, test_metrics_list=validation_metrics)
        results = ["PTQ Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in metrics.items()]
        logger.info("\n".join(results))

        return QuantizationResult(original_model=original_model, quantized_model=quantized_model, metrics=metrics)

    def qat(self, *args, **kwargs):
        raise NotImplementedError("QAT is not supported for OpenVinoQuantizer")

    def export(self, original_model, quantization_result, exporter):
        # TODO: Implement export
        return quantization_result


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

    logger.debug("Starting model quantization using NNCF without QC")
    calibration_dataset = nncf.Dataset(calibration_loader, transform_func=transform_fn_to_numpy)
    quantized_model = nncf.quantize(
        model,
        calibration_dataset=calibration_dataset,
        ignored_scope=ignored_scope,
        target_device=TargetDevice.CPU,
        subset_size=calibration_batches,  # TODO: Check whether subset_size is sample size or batch size
        preset=nncf.QuantizationPreset.MIXED,
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            # quantization_mode="symmetric",
            smooth_quant_alpha=-1,  # Not sure what it does, but it is present in Stable Diffusion V2 example
        ),
    )
    logger.debug("Model quantization using NNCF completed")
    return quantized_model
