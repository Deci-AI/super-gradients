import dataclasses
from typing import Union, Dict, Any

from torch import nn

from super_gradients.module_interfaces import PoseEstimationModelExportResult, ObjectDetectionModelExportResult

__all__ = ["QuantizationResult"]


@dataclasses.dataclass
class QuantizationResult:
    """
    :param original_model: The original model that came in to quantization function.
    :param quantized_model: The quantized model. The value may not be the same instance or have another class.
    :param metrics: The metrics of the quantized model computed on validation set.
    """

    original_model: nn.Module
    quantized_model: Union[nn.Module, Any]
    exported_model_path: Union[None, str]
    quantized_metrics: Dict[str, float]
    original_metrics: Dict[str, float]
    export_result: Union[None, ObjectDetectionModelExportResult, PoseEstimationModelExportResult] = None
