from typing import Tuple

import torch
from super_gradients.common.abstractions.abstract_logger import get_logger
from torch import nn, Tensor

logger = get_logger(__name__)


class OBBNMSAndReturnAsBatchedResult(nn.Module):
    __constants__ = ("batch_size", "confidence_threshold", "iou_threshold", "num_pre_nms_predictions", "max_predictions_per_image")

    def __init__(self, confidence_threshold: float, iou_threshold: float, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int):
        """
        Perform NMS on the output of the model and return the results in batched format.
        This module implements MatrixNMS algorithm for rotated bounding boxes.
        Hence, iou_threshold has different meaning compared to regular NMS.

        :param confidence_threshold:      The confidence threshold to apply to the model output
        :param iou_threshold:             The IoU threshold for selecting final detections.
               An iou_threshold has different meaning compared to regular NMS. In matrix NMS, it is the
               multiplication of predicted confidence score and decay factor for the bounding box (A decay applied to
               boxes that that has overlap with the current box).
        :param batch_size:                A fixed batch size for the model
        :param num_pre_nms_predictions:   The number of predictions before NMS step
        :param max_predictions_per_image: Maximum number of predictions per image
        """
        if max_predictions_per_image > num_pre_nms_predictions:
            raise ValueError(
                f"max_predictions_per_image ({max_predictions_per_image}) cannot be greater than num_pre_nms_predictions ({num_pre_nms_predictions})"
            )
        super().__init__()
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, input) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Take decoded predictions from the model, apply NMS to them and return the results in batched format.

        :param pred_boxes:       [B, N, 5] tensor, float32 in CXCYWHR format
        :param pred_scores:      [B, N, C] tensor, float32 class scores
        :return:                 A tuple of 4 tensors (num_detections, detection_boxes, detection_scores, detection_classes) will be returned:
                                 - A tensor of [batch_size, 1] containing the image indices for each detection.
                                 - A tensor of [batch_size, max_predictions_per_image, 5] containing the bounding box coordinates
                                   for each detection in [cx, cy, w, h, r] format.
                                 - A tensor of [batch_size, max_predictions_per_image] containing the confidence scores for each detection.
                                 - A tensor of [batch_size, max_predictions_per_image] containing the class indices for each detection.

        """
        from super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_post_prediction_callback import rboxes_matrix_nms

        pred_boxes, pred_scores = input
        pred_cls_conf, pred_cls_labels = torch.max(pred_scores, dim=2)

        # Apply confidence threshold
        pred_cls_conf = torch.masked_fill(pred_cls_conf, mask=pred_cls_conf < self.confidence_threshold, value=0)
        keep = rboxes_matrix_nms(pred_boxes, pred_cls_conf, iou_threshold=self.iou_threshold, already_sorted=True)

        num_predictions = []
        batched_pred_boxes = []
        batched_pred_scores = []
        batched_pred_classes = []
        for i in range(self.batch_size):
            keep_i = keep[i]
            pred_boxes_i = pred_boxes[keep_i]
            pred_scores_i = pred_cls_conf[keep_i]
            pred_classes_i = pred_cls_labels[keep_i]
            num_predictions_i = keep_i.sum()

            pad_size = self.max_predictions_per_image - pred_boxes.size(0)
            pred_boxes_i = torch.nn.functional.pad(pred_boxes_i, [0, 0, 0, pad_size], value=-1, mode="constant")
            pred_scores_i = torch.nn.functional.pad(pred_scores_i, [0, pad_size], value=-1, mode="constant")
            pred_classes_i = torch.nn.functional.pad(pred_classes_i, [0, pad_size], value=-1, mode="constant")

            num_predictions.append(num_predictions_i.reshape(1, 1))
            batched_pred_boxes.append(pred_boxes_i.unsqueeze(0))
            batched_pred_scores.append(pred_scores_i.unsqueeze(0))
            batched_pred_classes.append(pred_classes_i.unsqueeze(0))

        num_predictions = torch.cat(num_predictions, dim=0)
        batched_pred_boxes = torch.cat(batched_pred_boxes, dim=0)
        batched_pred_scores = torch.cat(batched_pred_scores, dim=0)
        batched_pred_classes = torch.cat(batched_pred_classes, dim=0)

        return num_predictions, batched_pred_boxes, batched_pred_scores, batched_pred_classes

    def get_output_names(self):
        return ["num_predictions", "pred_boxes", "pred_scores", "pred_classes"]

    def get_dynamic_axes(self):
        return {}


class OBBNMSAndReturnAsFlatResult(nn.Module):
    """
    Select the output from ONNX NMS node and return them in flat format.

    """

    __constants__ = ("iou_threshold", "confidence_threshold", "batch_size", "num_pre_nms_predictions", "max_predictions_per_image")

    def __init__(self, confidence_threshold, iou_threshold: float, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int):
        """
        Perform NMS on the output of the model and return the results in flat format.
        This module implements MatrixNMS algorithm for rotated bounding boxes.
        Hence, iou_threshold has different meaning compared to regular NMS.

        :param confidence_threshold:      The confidence threshold to apply to the model output
        :param iou_threshold:             The IoU threshold for selecting final detections.
               An iou_threshold has different meaning compared to regular NMS. In matrix NMS, it is the
               multiplication of predicted confidence score and decay factor for the bounding box (A decay applied to
               boxes that that has overlap with the current box).
        :param batch_size:                A fixed batch size for the model
        :param num_pre_nms_predictions:   The number of predictions before NMS step
        :param max_predictions_per_image: Maximum number of predictions per image
        """
        super().__init__()
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.max_predictions_per_image = max_predictions_per_image
        self.iou_threshold = iou_threshold

    def forward(self, input) -> Tensor:
        """
        Take decoded predictions from the model, apply NMS to them and return the results in flat format.

        :param pred_boxes:       [B, N, 5] tensor
        :param pred_scores:      [B, N, C] tensor
        :return:                 A single tensor of [Nout, 8] shape, where Nout is the total number of detections across all images in the batch.
                                 Each row will contain [image_index, cx, cy, w, h, r, class confidence, class index] values.
                                 Each image will have at most max_predictions_per_image detections.

        """
        from super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_post_prediction_callback import rboxes_matrix_nms

        pred_boxes, pred_scores = input
        dtype = pred_scores.dtype
        pred_cls_conf, pred_cls_labels = torch.max(pred_scores, dim=2)

        # Apply confidence threshold
        pred_cls_conf = torch.masked_fill(pred_cls_conf, mask=pred_cls_conf < self.confidence_threshold, value=0)
        keep = rboxes_matrix_nms(pred_boxes, pred_cls_conf, iou_threshold=self.iou_threshold, already_sorted=True)

        flat_results = []
        for i in range(self.batch_size):
            keep_i = keep[i]
            selected_boxes = pred_boxes[i][keep_i]
            selected_scores = pred_cls_conf[i][keep_i]
            label_indexes = pred_cls_labels[i][keep_i]
            batch_indexes = torch.full_like(label_indexes, i)

            flat_results_i = torch.cat(
                [batch_indexes.unsqueeze(-1).to(dtype), selected_boxes, selected_scores.unsqueeze(-1), label_indexes.unsqueeze(-1).to(dtype)], dim=1
            )
            flat_results_i = flat_results_i[: self.max_predictions_per_image]
            flat_results.append(flat_results_i)

        flat_results = torch.cat(flat_results, dim=0)
        return flat_results

    def get_output_names(self):
        return ["flat_predictions"]

    def get_dynamic_axes(self):
        return {
            "flat_predictions": {0: "num_predictions"},
        }
