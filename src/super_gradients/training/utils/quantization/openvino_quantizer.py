import copy
import dataclasses
from functools import partial
from typing import Optional, Mapping
from typing import Union, List

import nncf
from nncf import TargetDevice
from super_gradients.common.abstractions.abstract_logger import get_logger
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
class OpenVinoCalibrationParams:
    """ """

    num_calib_batches: int = 128


class OpenVinoQuantizer(AbstractQuantizer):
    def __init__(self, calibration_params: OpenVinoCalibrationParams, selective_quantization_params: OpenVinoSelectiveQuantizationParams):
        super().__init__()
        if isinstance(calibration_params, Mapping):
            calibration_params = OpenVinoCalibrationParams(**calibration_params)
        if isinstance(selective_quantization_params, Mapping):
            selective_quantization_params = OpenVinoSelectiveQuantizationParams(**selective_quantization_params)

        self.selective_quantization_params = selective_quantization_params
        self.calibration_params = calibration_params

    def ptq(
        self,
        model: nn.Module,
        trainer: "Trainer",
        calibration_loader: DataLoader,
        validation_loader: DataLoader,
        validation_metrics,
    ):
        model = copy.deepcopy(model)
        fuse_repvgg_blocks_residual_branches(model)

        quantized_model = openvino_ptq(
            model=model,
            calibration_loader=calibration_loader,
            calibration_batches=self.calibration_params.num_calib_batches,
            skip_patterns=self.selective_quantization_params.skip_patterns,
            skip_types=self.selective_quantization_params.skip_types,
        )

        # VALIDATE PTQ MODEL AND PRINT SUMMARY
        logger.info("Validating PTQ model...")
        metrics = trainer.test(model=quantized_model, test_loader=validation_loader, test_metrics_list=validation_metrics)
        results = ["PTQ Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in metrics.items()]
        logger.info("\n".join(results))

        return QuantizationResult(quantized_model=quantized_model, metrics=metrics)

    def qat(self, *args, **kwargs):
        raise NotImplementedError("QAT is not supported for OpenVinoQuantizer")


def openvino_ptq(
    model: nn.Module,
    calibration_loader,
    skip_patterns,
    skip_types,
    calibration_batches: int = 16,
):

    device = infer_model_device(model)

    def transform_fn_to_device(data_item, device):
        images = data_item[0]
        return images.to(device)

    def transform_fn_to_numpy(data_item):
        images = data_item[0]
        return images.numpy()

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
