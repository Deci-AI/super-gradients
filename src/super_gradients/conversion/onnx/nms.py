import os
import tempfile
from typing import Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxsim
import torch
from torch import nn, Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.conversion.onnx.utils import append_graphs

logger = get_logger(__name__)


class ConvertFlatTensorToTRTFormat(nn.Module):
    __constants__ = ("batch_size", "max_predictions")

    def __init__(self, batch_size: int, max_predictions_per_image: int):
        """

        :param batch_size: Fixed batch size (B)
        :param max_predictions_per_image: Fixed maximum number of predictions per image (N)
        """
        super().__init__()
        self.batch_size = batch_size
        self.max_predictions = max_predictions_per_image
        # This is the upper limit of predictions across all images in the batch
        self.total_max = self.max_predictions * self.batch_size

    def forward(self, predictions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Convert the predictions tensor in flat format to the format output by the TRT plugin

        :param predictions: [L, 7] tensor
            L - is the total number of predictions across all images in the batch
            Each row is [batch_index, x1, y1, x2, y2, score, class]

        :return: Tuple of tensors:
            - num_predictions: [B, 1] tensor (int64)
            - pred_boxes: [B, N, 4] tensor (float32)
            - pred_scores: [B, N] tensor (float32)
            - pred_classes: [B, N] tensor (int64)
        """
        predictions = torch.nn.functional.pad(predictions, (0, 0, 0, self.max_predictions * self.batch_size - predictions.size(0)), value=-1, mode="constant")

        batched_predictions = torch.zeros((self.batch_size, self.max_predictions, 6), dtype=predictions.dtype, device=predictions.device)

        batch_indexes = torch.arange(start=0, end=self.batch_size, step=1, device=predictions.device, dtype=predictions.dtype)
        masks = batch_indexes.view(-1, 1).eq(predictions[:, 0].view(1, -1))  # [B, N]

        num_predictions = torch.sum(masks, dim=1).long()

        for i in range(self.batch_size):
            selected_predictions = predictions[masks[i], 1:]
            batched_predictions[i] = torch.nn.functional.pad(
                selected_predictions, (0, 0, 0, self.max_predictions - selected_predictions.size(0)), value=0, mode="constant"
            )

        pred_boxes = batched_predictions[:, :, 0:4]
        pred_scores = batched_predictions[:, :, 4]
        pred_classes = batched_predictions[:, :, 5].long()

        return num_predictions.unsqueeze(1), pred_boxes, pred_scores, pred_classes

    @classmethod
    def as_graph(cls, batch_size: int, max_predictions_per_image) -> gs.Graph:
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_file = os.path.join(tmpdirname, "ConvertFlatTensorToTRTFormat.onnx")
            predictions = torch.zeros((batch_size * max_predictions_per_image // 2, 7))

            torch.onnx.export(
                ConvertFlatTensorToTRTFormat(batch_size=batch_size, max_predictions_per_image=max_predictions_per_image),
                args=predictions,
                f=onnx_file,
                input_names=["flat_predictions"],
                output_names=["num_predictions", "pred_boxes", "pred_scores", "pred_classes"],
                dynamic_axes={"flat_predictions": {0: "batch_size"}},
            )

            onnxsim.simplify(onnx_file)
            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            return convert_format_graph


class PickNMSPredictionsAndReturnAsBatchedResult(nn.Module):
    __constants__ = ("batch_size", "max_predictions_per_image")

    def __init__(self, batch_size: int, max_predictions_per_image: int):
        super().__init__()
        self.batch_size = batch_size
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, pred_boxes: Tensor, pred_scores: Tensor, selected_indexes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Select the predictions that are output by the NMS plugin.
        :param pred_boxes: [B, N, 4] tensor, float32
        :param pred_scores: [B, N, C] tensor, float32
        :param selected_indexes: [num_selected_indices, 3], int64 - each row is [batch_indexes, label_indexes, boxes_indexes]
        :return:

        """
        batch_indexes, label_indexes, boxes_indexes = selected_indexes[:, 0], selected_indexes[:, 1], selected_indexes[:, 2]

        selected_boxes = pred_boxes[batch_indexes, boxes_indexes]
        selected_scores = pred_scores[batch_indexes, boxes_indexes, label_indexes]

        predictions = torch.cat([batch_indexes.unsqueeze(1), selected_boxes, selected_scores.unsqueeze(1), label_indexes.unsqueeze(1)], dim=1)

        predictions = torch.nn.functional.pad(
            predictions, (0, 0, 0, self.max_predictions_per_image * self.batch_size - predictions.size(0)), value=-1, mode="constant"
        )

        batched_predictions = torch.zeros((self.batch_size, self.max_predictions_per_image, 6), dtype=predictions.dtype, device=predictions.device)

        batch_indexes = torch.arange(start=0, end=self.batch_size, step=1, device=predictions.device, dtype=predictions.dtype)
        masks = batch_indexes.view(-1, 1).eq(predictions[:, 0].view(1, -1))  # [B, N]

        num_predictions = torch.sum(masks, dim=1).long()

        for i in range(self.batch_size):
            selected_predictions = predictions[masks[i]]
            selected_predictions = selected_predictions[:, 1:]
            batched_predictions[i] = torch.nn.functional.pad(
                selected_predictions, (0, 0, 0, self.max_predictions_per_image - selected_predictions.size(0)), value=0, mode="constant"
            )

        pred_boxes = batched_predictions[:, :, 0:4]
        pred_scores = batched_predictions[:, :, 4]
        pred_classes = batched_predictions[:, :, 5].long()

        return num_predictions.unsqueeze(1), pred_boxes, pred_scores, pred_classes

    @classmethod
    def as_graph(cls, batch_size: int, max_predictions_per_image) -> gs.Graph:
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_file = os.path.join(tmpdirname, "PickNMSPredictionsAndReturnAsBatchedResult.onnx")
            pred_boxes = torch.zeros((batch_size, 300, 4))
            pred_scores = torch.zeros((batch_size, 300, 91))
            selected_indexes = torch.zeros((100, 3), dtype=torch.int64)

            torch.onnx.export(
                PickNMSPredictionsAndReturnAsBatchedResult(batch_size=batch_size, max_predictions_per_image=max_predictions_per_image),
                args=(pred_boxes, pred_scores, selected_indexes),
                f=onnx_file,
                input_names=["raw_boxes", "raw_scores", "selected_indexes"],
                output_names=["num_predictions", "pred_boxes", "pred_scores", "pred_classes"],
                dynamic_axes={
                    "raw_boxes": {
                        # 0: "batch_size",
                        1: "num_anchors"
                    },
                    "raw_scores": {
                        # 0: "batch_size",
                        1: "num_anchors",
                        2: "num_classes",
                    },
                    "selected_indexes": {0: "num_predictions"},
                },
            )

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            return convert_format_graph


class PickNMSPredictionsAndReturnAsFlatResult(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes: Tensor, pred_scores: Tensor, selected_indexes: Tensor):
        """
        Select the predictions that are output by the NMS plugin.
        :param pred_boxes: [B, N, 4] tensor
        :param pred_scores: [B, N, C] tensor
        :param selected_indexes: [num_selected_indices, 3] - each row is [batch_indexes, label_indexes, boxes_indexes]
        :return:
        """
        batch_indexes, label_indexes, boxes_indexes = selected_indexes[:, 0], selected_indexes[:, 1], selected_indexes[:, 2]

        selected_boxes = pred_boxes[batch_indexes, boxes_indexes]
        selected_scores = pred_scores[batch_indexes, boxes_indexes, label_indexes]

        return torch.cat([batch_indexes.unsqueeze(1), selected_boxes, selected_scores.unsqueeze(1), label_indexes.unsqueeze(1)], dim=1)

    @classmethod
    def as_graph(cls) -> gs.Graph:
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_file = os.path.join(tmpdirname, "PickNMSPredictionsAndReturnAsFlatTensor.onnx")
            pred_boxes = torch.zeros((2, 100, 4))
            pred_scores = torch.zeros((2, 100, 91))

            torch.onnx.export(
                PickNMSPredictionsAndReturnAsFlatResult(),
                args=(pred_boxes, pred_scores, torch.zeros((100, 3), dtype=torch.int64)),
                f=onnx_file,
                input_names=["pred_boxes", "pred_scores", "selected_indexes"],
                output_names=["flat_predictions"],
                dynamic_axes={
                    "pred_boxes": {0: "batch_size", 1: "num_anchors"},
                    "pred_scores": {0: "batch_size", 1: "num_anchors", 2: "num_classes"},
                    "selected_indexes": {0: "num_predictions"},
                    "flat_predictions": {0: "num_predictions"},
                },
            )

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            return convert_format_graph


def attach_onnx_nms(
    onnx_model_path: str,
    output_onnx_model_path,
    max_predictions_per_image: int,
    confidence_threshold: float,
    nms_threshold: float,
    batch_size: int,
    output_predictions_format: str,
):
    """
    Attach ONNX NMS plugin to the detection model.
    The model should have exactly two outputs: pred_boxes and pred_scores.
        - pred_boxes: [batch_size, num_anchors, 4]
        - pred_scores: [batch_size, num_anchors, num_classes]
    This function will add the NMS layer to the model and return predictions in the format defined by output_format.

    :param onnx_model_path: Input ONNX model path
    :param output_onnx_model_path: Output ONNX model path. Can be the same as input model path.
    :param batch_size: The batch size used for the inference.
    :param max_predictions_per_image: Maximum number of predictions per image
    :param confidence_threshold: The confidence threshold to use for detections.
    :param nms_threshold: The NMS threshold to use for detections.
    :param output_predictions_format: The output format of the predictions. Can be "flat" or "batched".

    If output_format equals to "flat":
    A single tensor of [N, 7] will be returned, where N is the total number of detections across all images in the batch.
    Each row will contain [image_index, x1, y1, x2, y2, class_index, confidence].

    If output_format equals to "batched" format:
    A tuple of 4 tensors (num_detections, detection_boxes, detection_scores, detection_classes) will be returned:
    - A tensor of [batch_size, 1] containing the image indices for each detection.
    - A tensor of [batch_size, max_output_boxes, 4] containing the bounding box coordinates for each detection in [x1, y1, x2, y2] format.
    - A tensor of [batch_size, max_output_boxes] containing the confidence scores for each detection.
    - A tensor of [batch_size, max_output_boxes] containing the class indices for each detection.

    :return: None
    """
    graph = gs.import_onnx(onnx.load(onnx_model_path))
    graph.fold_constants()

    pred_boxes, pred_scores = graph.outputs

    permute_scores = gs.Variable(
        name="permuted_scores",
        dtype=np.float32,
    )
    graph.layer(op="Transpose", name="permute_scores", inputs=[pred_scores], outputs=[permute_scores], attrs={"perm": [0, 2, 1]})

    op_inputs = [pred_boxes, permute_scores] + [
        gs.Constant(name="max_output_boxes_per_class", values=np.array([max_predictions_per_image], dtype=np.int64)),
        gs.Constant(name="iou_threshold", values=np.array([nms_threshold], dtype=np.float32)),
        gs.Constant(name="score_threshold", values=np.array([confidence_threshold], dtype=np.float32)),
    ]
    logger.debug(f"op_inputs: {op_inputs}")

    # NMS Outputs
    # selected indices from the boxes tensor. [num_selected_indices, 3], the selected index format is [batch_index, class_index, box_index].
    output_selected_indices = gs.Variable(
        name="selected_indices",
        dtype=np.int64,
        shape=["num_selected_indices", 3],
    )  # A scalar indicating the number of valid detections per batch image.

    # Create the NMS Plugin node with the selected inputs. The outputs of the node will also
    # become the final outputs of the graph.
    graph.layer(
        op="NonMaxSuppression",
        name="batched_nms",
        inputs=op_inputs,
        outputs=[output_selected_indices],
        attrs={
            "center_point_box": 0,
        },
    )

    graph.outputs = [pred_boxes, pred_boxes, output_selected_indices]

    if output_predictions_format == "batched":
        convert_format_graph = PickNMSPredictionsAndReturnAsBatchedResult.as_graph(batch_size=batch_size, max_predictions_per_image=max_predictions_per_image)
        graph = append_graphs(graph, convert_format_graph)
    elif output_predictions_format == "flat":
        convert_format_graph = PickNMSPredictionsAndReturnAsFlatResult.as_graph()
        graph = append_graphs(graph, convert_format_graph)
    else:
        raise ValueError(f"Invalid output_predictions_format: {output_predictions_format}")

    # Final cleanup & save
    graph.cleanup().toposort()

    # iteratively_infer_shapes(graph)

    model = gs.export_onnx(graph)
    onnx.save(model, output_onnx_model_path)
    logger.debug(f"Saved ONNX model to {output_onnx_model_path}")

    # onnxsim.simplify(output_onnx_model_path)
    # logger.debug(f"Ran onnxsim.simplify on {output_onnx_model_path}")
