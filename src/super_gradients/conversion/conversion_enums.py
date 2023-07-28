from enum import Enum

__all__ = ["DetectionOutputFormatMode", "ExportQuantizationMode", "ExportTargetBackend"]


class ExportTargetBackend(str, Enum):
    """Enum for specifying target backend for exporting a model."""

    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"


class DetectionOutputFormatMode(str, Enum):
    """Enum for specifying output format for the detection model when postprocessing & NMS is enabled."""

    FLAT_FORMAT = "flat"
    BATCH_FORMAT = "batch"


class ExportQuantizationMode(str, Enum):
    """Enum for specifying quantization mode."""

    FP16 = "fp16"
    INT8 = "int8"
