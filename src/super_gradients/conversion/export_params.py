import dataclasses
from typing import Optional, Tuple

from super_gradients.conversion.conversion_enums import ExportTargetBackend, DetectionOutputFormatMode


@dataclasses.dataclass
class ExportParams:
    """
    Parameters for exporting a model to ONNX format in PTQ/QAT methods of Trainer.
    Most of the parameters are related ot ExportableObjectDetectionModel.export method.

    :param output_onnx_path: The path to save the ONNX model.
           If None, the ONNX filename will use current experiment dir folder
           and the output filename will reflect model input shape & whether it's a PTQ or QAT model.

    :param batch_size: The batch size for the ONNX model. Default is 1.

    :param input_image_shape: The input image shape (rows, cols) for the ONNX model.
           If None, the input shape will be inferred from the model or validation dataloader.

    :param preprocessing: If True, the preprocessing will be included in the ONNX model.
           This option is only available for models that support model.export() syntax.

    :param postprocessing: If True, the postprocessing will be included in the ONNX model.
           This option is only available for models that support model.export() syntax.

    :param confidence_threshold: The confidence threshold for object detection models
           or image binary segmentation models.
           This attribute used only for models inheriting ExportableSegmentationModel
           and ExportableObjectDetectionModel.
           If None, the default confidence threshold for a given model will be used.
    :param onnx_export_kwargs: (dict) Optional keyword arguments for torch.onnx.export() function.
    :param onnx_simplify: (bool) If True, apply onnx-simplifier to the exported model.

    :param detection_nms_iou_threshold: (float) A IoU threshold for the NMS step.
           Relevant only for object detection models and only if postprocessing is True.
           Default to None, which means the default value for a given model will be used.

    :param detection_max_predictions_per_image: (int) Maximum number of predictions per image.
           Relevant only for object detection models and only if postprocessing is True.

    :param detection_num_pre_nms_predictions: (int) Number of predictions to keep before NMS.
           Relevant only for object detection models and only if postprocessing is True.

    :param detection_predictions_format: (DetectionOutputFormatMode) Format of the output predictions of detection models.
           Possible values:
           DetectionOutputFormatMode.BATCH_FORMAT - A tuple of 4 tensors will be returned
           (num_detections, detection_boxes, detection_scores, detection_classes)
           - A tensor of [batch_size, 1] containing the image indices for each detection.
           - A tensor of [batch_size, max_output_boxes, 4] containing the bounding box coordinates for each detection in [x1, y1, x2, y2] format.
           - A tensor of [batch_size, max_output_boxes] containing the confidence scores for each detection.
           - A tensor of [batch_size, max_output_boxes] containing the class indices for each detection.

           DetectionOutputFormatMode.FLAT_FORMAT - Tensor of shape [N, 7], where N is the total number of
           predictions in the entire batch.
           Each row will contain [image_index, x1, y1, x2, y2, class confidence, class index] values.
           Relevant only for object detection models and only if postprocessing is True.
    """

    output_onnx_path: Optional[str] = None
    engine: Optional[ExportTargetBackend] = None
    batch_size: int = 1
    input_image_shape: Optional[Tuple[int, int]] = None
    preprocessing: bool = True
    postprocessing: bool = True
    confidence_threshold: Optional[float] = None

    onnx_export_kwargs: Optional[dict] = None
    onnx_simplify: bool = True

    # These are only relevant for object detection and pose estimation models
    detection_nms_iou_threshold: Optional[float] = None
    detection_max_predictions_per_image: Optional[int] = None
    detection_predictions_format: DetectionOutputFormatMode = DetectionOutputFormatMode.BATCH_FORMAT
    detection_num_pre_nms_predictions: int = 1000
