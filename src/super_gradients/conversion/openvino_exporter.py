from functools import partial
from typing import Any, Union

from super_gradients.conversion.abstract_exporter import AbstractExporter
import nncf
import openvino as ov
from pathlib import Path
import torch


class OpenVINOExporter(AbstractExporter):
    def __init__(self, *, output_file: str, example_input: Any):
        self.output_file = output_file
        self.example_input = example_input

    def export_fp32(self, model: Union[torch.nn.Module, str, Path, ov.Model]):
        ov_model = self._get_ov_model_from_input_model(model)
        ov.save_model(ov_model, self.output_file, compress_to_fp16=False)

    def export_fp16(self, model: torch.nn.Module):
        ov_model = self._get_ov_model_from_input_model(model)
        ov.save_model(ov_model, self.output_file, compress_to_fp16=True)

    def _get_ov_model_from_input_model(self, model) -> ov.Model:
        if isinstance(model, torch.nn.Module):
            ov_model = ov.convert_model(model, input=self.example_input)
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

    def _export_int8_non_calibrated(self, model: torch.nn.Module, calibration_dataset):
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
