import dataclasses
from typing import Union, Dict

from torch import nn

from super_gradients.module_interfaces import PoseEstimationModelExportResult, ObjectDetectionModelExportResult

__all__ = ["QuantizationResult"]


@dataclasses.dataclass
class QuantizationResult:
    quantized_model: nn.Module
    output_onnx_path: str
    valid_metrics_dict: Dict[str, float]
    export_result: Union[None, ObjectDetectionModelExportResult, PoseEstimationModelExportResult] = None
