from functools import partial
from typing import Any, Union

from super_gradients.common.registry.registry import register_exporter
from super_gradients.conversion.abstract_exporter import AbstractExporter
import nncf
import openvino as ov
from pathlib import Path
import torch


@register_exporter()
class OpenVinoExporter(AbstractExporter):
    def __init__(self, *, output_path: str):
        self.output_path = output_path
        # self.example_input = example_input

    def export_fp32(self, model: Union[torch.nn.Module, str, Path, ov.Model]):
        ov_model = self._get_ov_model_from_input_model(model)
        ov.save_model(ov_model, self.output_path, compress_to_fp16=False)

    def export_fp16(self, model: torch.nn.Module):
        ov_model = self._get_ov_model_from_input_model(model)
        ov.save_model(ov_model, self.output_path, compress_to_fp16=True)

    def export_quantized(self, original_model, quantized_model):
        ov_model = self._get_ov_model_from_input_model(quantized_model)
        ov.save_model(ov_model, self.output_path, compress_to_fp16=False)

    def _get_ov_model_from_input_model(self, model, example_input=None) -> ov.Model:
        if isinstance(model, torch.nn.Module):
            if example_input is None:
                raise ValueError("Model conversion from PyTorch requires example input")
            ov_model = ov.convert_model(model, input=example_input)
        elif isinstance(model, (str, Path)):
            ov_model = ov.convert_model(str(model))
        elif isinstance(model, ov.Model):
            ov_model = model
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        return ov_model

    def export_int8(self, model: Union[torch.nn.Module, ov.Model], calibration_dataset):
        if isinstance(model, torch.nn.Module):
            return self._export_int8_non_calibrated(model, calibration_dataset)
        else:
            return self._export_int8_already_calibrated(model, calibration_dataset)

    def _export_int8_already_calibrated(self, model: ov.Model, calibration_dataset):
        pass

    def _export_int8_non_calibrated(self, model: torch.nn.Module, calibration_loader):
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
