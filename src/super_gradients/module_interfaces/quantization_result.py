import dataclasses
from typing import Union, Dict, Any

from torch import nn
from torch.utils.data import DataLoader

# from super_gradients.module_interfaces import PoseEstimationModelExportResult, ObjectDetectionModelExportResult, SegmentationModelExportResult

__all__ = ["QuantizationResult"]


@dataclasses.dataclass
class QuantizationResult:
    """
    :param original_model: The original model that came in to quantization function.
    :param quantized_model: The quantized model. The value may not be the same instance or have another class.
    :param original_metrics: The metrics of the original model computed on validation set.
    :param quantized_metrics: The metrics of the quantized model computed on validation set.
    :param export_path: The path to the exported model. If the model was not exported, the value is None.
    :param export_result: The result of the export operation. If the model was not exported, the value is None.
    """

    original_model: nn.Module
    original_metrics: Dict[str, float]

    quantized_model: Union[nn.Module, Any]
    quantized_metrics: Dict[str, float]

    calibration_dataloader: DataLoader

    export_path: Union[None, str]
    export_result: Union[None, Any] = None
