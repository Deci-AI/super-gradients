from enum import Enum

__all__ = ["DetectionOutputFormatMode", "ExportQuantizationMode", "ExportTargetBackend"]


class ExportTargetBackend(str, Enum):
    """Enum for specifying target backend for exporting a model."""

    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"


class DetectionOutputFormatMode(str, Enum):
    """Enum for specifying output format for the detection model when postprocessing & NMS is enabled.

    :attr:`FLAT_FORMAT`:
        Predictions is a single tensor of shape [N, 7].
        N is the total number of detections in the entire batch.
        Each row will contain [image_index, x1, y1, x2, y2, class confidence, class index] values.

    :attr:`BATCH_FORMAT`:
        A tuple of 4 tensors (num_detections, detection_boxes, detection_scores, detection_classes) will be returned:
        - A tensor of [batch_size, 1] containing the image indices for each detection.
        - A tensor of [batch_size, max_output_boxes, 4] containing the bounding box coordinates for each detection in [x1, y1, x2, y2] format.
        - A tensor of [batch_size, max_output_boxes] containing the confidence scores for each detection.
        - A tensor of [batch_size, max_output_boxes] containing the class indices for each detection.
    """

    FLAT_FORMAT = "flat"
    BATCH_FORMAT = "batch"


class ExportQuantizationMode(str, Enum):
    """Enum for specifying quantization mode."""

    FP16 = "fp16"
    INT8 = "int8"
