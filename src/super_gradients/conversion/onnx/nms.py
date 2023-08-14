import os
import tempfile
from typing import Tuple

import numpy as np
import onnx
import onnx.shape_inference
import torch
from onnx import TensorProto
from torch import nn, Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.conversion.conversion_enums import DetectionOutputFormatMode
from super_gradients.conversion.conversion_utils import numpy_dtype_to_torch_dtype
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_fail_with_instructions
from super_gradients.conversion.onnx.utils import append_graphs

logger = get_logger(__name__)

gs = import_onnx_graphsurgeon_or_fail_with_instructions()


class PickNMSPredictionsAndReturnAsBatchedResult(nn.Module):
    __constants__ = ("batch_size", "max_predictions_per_image")

    def __init__(self, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, pred_boxes: Tensor, pred_scores: Tensor, selected_indexes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Select the predictions that are output by the NMS plugin.
        :param pred_boxes: [B, N, 4] tensor, float32
        :param pred_scores: [B, N, C] tensor, float32
        :param selected_indexes: [num_selected_indices, 3], int64 - each row is [batch_indexes, label_indexes, boxes_indexes]
        :return: A tuple of 4 tensors (num_detections, detection_boxes, detection_scores, detection_classes) will be returned:
            - A tensor of [batch_size, 1] containing the image indices for each detection.
            - A tensor of [batch_size, max_output_boxes, 4] containing the bounding box coordinates for each detection in [x1, y1, x2, y2] format.
            - A tensor of [batch_size, max_output_boxes] containing the confidence scores for each detection.
            - A tensor of [batch_size, max_output_boxes] containing the class indices for each detection.

        """
        batch_indexes, label_indexes, boxes_indexes = selected_indexes[:, 0], selected_indexes[:, 1], selected_indexes[:, 2]

        selected_boxes = pred_boxes[batch_indexes, boxes_indexes]
        selected_scores = pred_scores[batch_indexes, boxes_indexes, label_indexes]

        predictions = torch.cat([batch_indexes.unsqueeze(1), selected_boxes, selected_scores.unsqueeze(1), label_indexes.unsqueeze(1)], dim=1)

        predictions = torch.nn.functional.pad(
            predictions, (0, 0, 0, self.max_predictions_per_image * self.batch_size - predictions.size(0)), value=-1, mode="constant"
        )

        batch_predictions = torch.zeros((self.batch_size, self.max_predictions_per_image, 6), dtype=predictions.dtype, device=predictions.device)

        batch_indexes = torch.arange(start=0, end=self.batch_size, step=1, device=predictions.device).to(dtype=predictions.dtype)
        masks = batch_indexes.view(-1, 1).eq(predictions[:, 0].view(1, -1))  # [B, N]

        num_predictions = torch.sum(masks, dim=1).long()

        for i in range(self.batch_size):
            selected_predictions = predictions[masks[i]]
            selected_predictions = selected_predictions[:, 1:]
            batch_predictions[i] = torch.nn.functional.pad(
                selected_predictions, (0, 0, 0, self.max_predictions_per_image - selected_predictions.size(0)), value=0, mode="constant"
            )

        pred_boxes = batch_predictions[:, :, 0:4]
        pred_scores = batch_predictions[:, :, 4]
        pred_classes = batch_predictions[:, :, 5].long()

        return num_predictions.unsqueeze(1), pred_boxes, pred_scores, pred_classes

    @classmethod
    def as_graph(cls, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image, dtype: torch.dtype, device: torch.device) -> gs.Graph:
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_file = os.path.join(tmpdirname, "PickNMSPredictionsAndReturnAsBatchedResult.onnx")
            pred_boxes = torch.zeros((batch_size, num_pre_nms_predictions, 4), dtype=dtype, device=device)
            pred_scores = torch.zeros((batch_size, num_pre_nms_predictions, 3), dtype=dtype, device=device)
            selected_indexes = torch.zeros((max_predictions_per_image, 3), dtype=torch.int64, device=device)

            torch.onnx.export(
                PickNMSPredictionsAndReturnAsBatchedResult(
                    batch_size=batch_size, num_pre_nms_predictions=num_pre_nms_predictions, max_predictions_per_image=max_predictions_per_image
                ).to(device=device, dtype=dtype),
                args=(pred_boxes, pred_scores, selected_indexes),
                f=onnx_file,
                input_names=["raw_boxes", "raw_scores", "selected_indexes"],
                output_names=["num_predictions", "pred_boxes", "pred_scores", "pred_classes"],
                dynamic_axes={
                    "raw_boxes": {
                        # 0: "batch_size",
                        # 1: "num_anchors"
                    },
                    "raw_scores": {
                        # 0: "batch_size",
                        # 1: "num_anchors",
                        2: "num_classes",
                    },
                    "selected_indexes": {0: "num_predictions"},
                },
            )

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            return convert_format_graph


class PickNMSPredictionsAndReturnAsFlatResult(nn.Module):
    __constants__ = ("batch_size", "num_pre_nms_predictions", "max_predictions_per_image")

    def __init__(self, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, pred_boxes: Tensor, pred_scores: Tensor, selected_indexes: Tensor):
        """
        Select the predictions that are output by the NMS plugin.
        :param pred_boxes: [B, N, 4] tensor
        :param pred_scores: [B, N, C] tensor
        :param selected_indexes: [num_selected_indices, 3] - each row is [batch_indexes, label_indexes, boxes_indexes]
        :return: A single tensor of [Nout, 7] shape, where Nout is the total number of detections across all images in the batch.
        Each row will contain [image_index, x1, y1, x2, y2, class confidence, class index] values.

        """
        batch_indexes, label_indexes, boxes_indexes = selected_indexes[:, 0], selected_indexes[:, 1], selected_indexes[:, 2]

        selected_boxes = pred_boxes[batch_indexes, boxes_indexes]
        selected_scores = pred_scores[batch_indexes, boxes_indexes, label_indexes]

        return torch.cat(
            [
                batch_indexes.unsqueeze(1).to(selected_boxes.dtype),
                selected_boxes,
                selected_scores.unsqueeze(1),
                label_indexes.unsqueeze(1).to(selected_boxes.dtype),
            ],
            dim=1,
        )

    @classmethod
    def as_graph(cls, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int, dtype: torch.dtype, device: torch.device) -> gs.Graph:
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_file = os.path.join(tmpdirname, "PickNMSPredictionsAndReturnAsFlatTensor.onnx")
            pred_boxes = torch.zeros((batch_size, num_pre_nms_predictions, 4), dtype=dtype, device=device)
            pred_scores = torch.zeros((batch_size, num_pre_nms_predictions, 91), dtype=dtype, device=device)
            selected_indexes = torch.zeros((max_predictions_per_image // 2, 3), dtype=torch.int64, device=device)

            torch.onnx.export(
                PickNMSPredictionsAndReturnAsFlatResult(
                    batch_size=batch_size, num_pre_nms_predictions=num_pre_nms_predictions, max_predictions_per_image=max_predictions_per_image
                ),
                args=(pred_boxes, pred_scores, selected_indexes),
                f=onnx_file,
                input_names=["pred_boxes", "pred_scores", "selected_indexes"],
                output_names=["flat_predictions"],
                dynamic_axes={
                    "pred_boxes": {},
                    "pred_scores": {2: "num_classes"},
                    "selected_indexes": {0: "num_predictions"},
                    "flat_predictions": {0: "num_predictions"},
                },
            )

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            return convert_format_graph


def attach_onnx_nms(
    onnx_model_path: str,
    output_onnx_model_path,
    num_pre_nms_predictions: int,
    max_predictions_per_image: int,
    confidence_threshold: float,
    nms_threshold: float,
    batch_size: int,
    output_predictions_format: DetectionOutputFormatMode,
    device: torch.device,
):
    """
    Attach ONNX NMS plugin to the detection model.
    The model should have exactly two outputs: pred_boxes and pred_scores.
        - pred_boxes: [batch_size, num_pre_nms_predictions, 4]
        - pred_scores: [batch_size, num_pre_nms_predictions, num_classes]
    This function will add the NMS layer to the model and return predictions in the format defined by output_format.

    :param onnx_model_path: Input ONNX model path
    :param output_onnx_model_path: Output ONNX model path. Can be the same as input model path.
    :param num_pre_nms_predictions:
    :param batch_size: The batch size used for the inference.
    :param max_predictions_per_image: Maximum number of predictions per image
    :param confidence_threshold: The confidence threshold to use for detections.
    :param nms_threshold: The NMS threshold to use for detections.
    :param output_predictions_format: The output format of the predictions. Can be "flat" or "batch".

    If output_format equals to "flat":
    A single tensor of [N, 7] will be returned, where N is the total number of detections across all images in the batch.
    Each row will contain [image_index, x1, y1, x2, y2, confidence, class_index].

    If output_format equals to "batch" format:
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

    graph_output_dtype = pred_scores.dtype

    if graph_output_dtype == np.float16:
        pred_scores_f32 = gs.Variable(
            name="pred_scores_f32",
            dtype=np.float32,
            shape=pred_scores.shape,
        )
        pred_boxes_f32 = gs.Variable(
            name="pred_boxes_f32",
            dtype=np.float32,
            shape=pred_boxes.shape,
        )
        graph.layer(op="Cast", name="cast_boxes_to_fp32", inputs=[pred_boxes], outputs=[pred_boxes_f32], attrs={"to": TensorProto.FLOAT})
        graph.layer(op="Cast", name="cast_scores_to_fp32", inputs=[pred_scores], outputs=[pred_scores_f32], attrs={"to": TensorProto.FLOAT})

        pred_scores = pred_scores_f32
        pred_boxes = pred_boxes_f32
    elif graph_output_dtype == np.float32:
        pass
    else:
        raise ValueError(f"Invalid dtype: {graph_output_dtype}")

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

    graph.outputs = [pred_boxes, pred_scores, output_selected_indices]

    if output_predictions_format == DetectionOutputFormatMode.BATCH_FORMAT:
        convert_format_graph = PickNMSPredictionsAndReturnAsBatchedResult.as_graph(
            batch_size=batch_size,
            num_pre_nms_predictions=num_pre_nms_predictions,
            max_predictions_per_image=max_predictions_per_image,
            dtype=numpy_dtype_to_torch_dtype(np.float32),
            device=device,
        )
        graph = append_graphs(graph, convert_format_graph)
    elif output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
        convert_format_graph = PickNMSPredictionsAndReturnAsFlatResult.as_graph(
            batch_size=batch_size,
            num_pre_nms_predictions=num_pre_nms_predictions,
            max_predictions_per_image=max_predictions_per_image,
            dtype=numpy_dtype_to_torch_dtype(np.float32),
            device=device,
        )
        graph = append_graphs(graph, convert_format_graph)
    else:
        raise ValueError(f"Invalid output_predictions_format: {output_predictions_format}")

    # Final cleanup & save
    graph.cleanup().toposort()

    # iteratively_infer_shapes(graph)

    model = gs.export_onnx(graph)
    onnx.shape_inference.infer_shapes(model)
    onnx.save(model, output_onnx_model_path)
    logger.debug(f"Saved ONNX model to {output_onnx_model_path}")

    # onnxsim.simplify(output_onnx_model_path)
    # logger.debug(f"Ran onnxsim.simplify on {output_onnx_model_path}")
