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

__all__ = ["attach_onnx_pose_nms"]


class PoseNMSAndReturnAsBatchedResult(nn.Module):
    __constants__ = ("batch_size", "max_predictions_per_image")

    def __init__(self, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int):
        """

        :param batch_size:                The batch size used for the inference. Since current export does not support dynamic batch size,
                                          this value must be known at export time.
        :param num_pre_nms_predictions:   The number of predictions before NMS step (per image).
                                          Usually it is less than total number of predictions that model outputs
                                          and top-K predictions are selected (based on score).
        :param max_predictions_per_image: The number of predictions after NMS step (per image). If after NMS less than
                                          max_predictions_per_image predictions are left,
                                          the rest of the predictions will be padded with 0.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, pred_boxes: Tensor, pred_scores: Tensor, pred_joints: Tensor, selected_indexes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Select the predictions that are output by the NMS plugin.

        :param pred_boxes: [B, N, 4] tensor, float32 in XYXY format
        :param pred_scores: [B, N, 1] tensor, float32
        :param pred_joints: [B, N, Num Joints, 3] tensor, float32
        :param selected_indexes: [num_selected_indices, 3], int64 - each row is [batch_indexes, label_indexes, boxes_indexes]

        :return: A tuple of 4 tensors (num_detections, boxes, scores, joints) will be returned:
            - A tensor of [batch_size, 1] containing the image indices for each detection.
            - A tensor of [batch_size, max_output_boxes, 4] containing the bounding box coordinates for each detection in [x1, y1, x2, y2] format.
            - A tensor of [batch_size, max_output_boxes, Num Joints, 3]

        """
        batch_indexes, label_indexes, boxes_indexes = selected_indexes[:, 0], selected_indexes[:, 1], selected_indexes[:, 2]

        selected_boxes = pred_boxes[batch_indexes, boxes_indexes]  # [num_detections, 4]
        selected_scores = pred_scores[batch_indexes, boxes_indexes, label_indexes]  # [num_detections]
        selected_poses = pred_joints[batch_indexes, boxes_indexes]  # [num_detections, Num Joints, 3]

        predictions = torch.cat([batch_indexes.unsqueeze(1), selected_boxes, selected_scores.unsqueeze(1), selected_poses.flatten(1)], dim=1)

        predictions = torch.nn.functional.pad(
            predictions, (0, 0, 0, self.max_predictions_per_image * self.batch_size - predictions.size(0)), value=-1, mode="constant"
        )

        batch_predictions = torch.zeros(
            (self.batch_size, self.max_predictions_per_image, 4 + 1 + selected_poses.size(1) * selected_poses.size(2)),
            dtype=predictions.dtype,
            device=predictions.device,
        )

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
        pred_joints = batch_predictions[:, :, 5:].reshape(self.batch_size, self.max_predictions_per_image, -1, 3)

        return num_predictions.unsqueeze(1), pred_boxes, pred_scores, pred_joints

    @classmethod
    def as_graph(cls, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int, dtype: torch.dtype, device: torch.device) -> gs.Graph:
        """
        Convert this module to a separate ONNX graph in order to attach it to the main model.

        :param batch_size:                The batch size used for the inference. Since current export does not support dynamic batch size,
                                          this value must be known at export time.
        :param num_pre_nms_predictions:   The number of predictions before NMS step (per image).
                                          Usually it is less than total number of predictions that model outputs and top-K
                                          predictions are selected (based on score).
        :param max_predictions_per_image: The number of predictions after NMS step (per image). If after NMS less than
                                          max_predictions_per_image predictions are left, the rest of the predictions will be padded with 0.
        :param dtype:                     The target dtype for the graph. If user asked for FP16 model we should create underlying graph with FP16 tensors.
        :param device:                    The target device for exporting graph.
        :return:                          An instance of GraphSurgeon graph that can be attached to the main model.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_file = os.path.join(tmpdirname, "PoseNMSAndReturnAsBatchedResult.onnx")
            pre_nms_boxes = torch.zeros((batch_size, num_pre_nms_predictions, 4), dtype=dtype, device=device)
            pre_nms_scores = torch.zeros((batch_size, num_pre_nms_predictions, 1), dtype=dtype, device=device)
            pre_nms_joints = torch.zeros((batch_size, num_pre_nms_predictions, 17, 3), dtype=dtype, device=device)
            selected_indexes = torch.zeros((max_predictions_per_image, 3), dtype=torch.int64, device=device)

            torch.onnx.export(
                PoseNMSAndReturnAsBatchedResult(
                    batch_size=batch_size, num_pre_nms_predictions=num_pre_nms_predictions, max_predictions_per_image=max_predictions_per_image
                ).to(device=device, dtype=dtype),
                args=(pre_nms_boxes, pre_nms_scores, pre_nms_joints, selected_indexes),
                f=onnx_file,
                input_names=["input_pre_nms_boxes", "input_pre_nms_scores", "input_pre_nms_joints", "selected_indexes"],
                output_names=["num_predictions", "post_nms_boxes", "post_nms_scores", "post_nms_joints"],
                dynamic_axes={
                    "input_pre_nms_boxes": {
                        # 0: "batch_size",
                        # 1: "num_anchors"
                    },
                    "input_pre_nms_scores": {
                        # 0: "batch_size",
                        # 1: "num_anchors",
                    },
                    "input_pre_nms_joints": {
                        # 0: "batch_size",
                        # 1: "num_anchors",
                        # We can make this static, although it would complicate the code as one would have to pass the number of joints to the function
                        2: "num_joints",
                    },
                    "selected_indexes": {0: "num_predictions"},
                },
            )

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            return convert_format_graph


class PoseNMSAndReturnAsFlatResult(nn.Module):
    __constants__ = ("batch_size", "num_pre_nms_predictions", "max_predictions_per_image")

    def __init__(self, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int):
        """

        :param batch_size:                The batch size used for the inference. Since current export does not support dynamic batch size,
                                          this value must be known at export time.
        :param num_pre_nms_predictions:   The number of predictions before NMS step (per image).
                                          Usually it is less than total number of predictions that model outputs and
                                          top-K predictions are selected (based on score).
        :param max_predictions_per_image: Not used, exists for compatibility with PoseNMSAndReturnAsBatchedResult
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, pred_boxes: Tensor, pred_scores: Tensor, pred_joints: Tensor, selected_indexes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Select the predictions that are output by the NMS plugin.

        :param pred_boxes: [B, N, 4] tensor, float32
        :param pred_scores: [B, N, 1] tensor, float32
        :param pred_joints: [B, N, Num Joints, 3] tensor, float32
        :param selected_indexes: [num_selected_indices, 3], int64 - each row is [batch_indexes, label_indexes, boxes_indexes]

        :return: A single tensor of [Nout, 7] shape, where Nout is the total number of detections across all images in the batch.
        Each row will contain [image_index, x1, y1, x2, y2, class confidence, class index] values.

        """
        batch_indexes, label_indexes, boxes_indexes = selected_indexes[:, 0], selected_indexes[:, 1], selected_indexes[:, 2]

        selected_boxes = pred_boxes[batch_indexes, boxes_indexes]  # [num_detections, 4]
        selected_scores = pred_scores[batch_indexes, boxes_indexes, label_indexes].unsqueeze(1)  # [num_detections, 1]
        selected_poses = pred_joints[batch_indexes, boxes_indexes].flatten(start_dim=1)  # [num_detections, (Num Joints * 3)]

        return torch.cat(
            [
                batch_indexes.unsqueeze(1).to(selected_boxes.dtype),
                selected_boxes,
                selected_scores,
                selected_poses,
            ],
            dim=1,
        )

    @classmethod
    def as_graph(cls, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int, dtype: torch.dtype, device: torch.device) -> gs.Graph:
        """
        Convert this module to a separate ONNX graph in order to attach it to the main model.

        :param batch_size:                The batch size used for the inference. Since current export does not support dynamic batch size,
                                          this value must be known at export time.
        :param num_pre_nms_predictions:   The number of predictions before NMS step (per image).
                                          Usually it is less than total number of predictions that model outputs and
                                          top-K predictions are selected (based on score).
        :param max_predictions_per_image: Not used, exists for compatibility with PoseNMSAndReturnAsBatchedResult
        :param dtype:                     The target dtype for the graph. If user asked for FP16 model we should create underlying graph with FP16 tensors.
        :param device:                    The target device for exporting graph.
        :return:                          An instance of GraphSurgeon graph that can be attached to the main model.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_file = os.path.join(tmpdirname, "PoseNMSAndReturnAsFlatResult.onnx")
            pre_nms_boxes = torch.zeros((batch_size, num_pre_nms_predictions, 4), dtype=dtype, device=device)
            pre_nms_scores = torch.zeros((batch_size, num_pre_nms_predictions, 1), dtype=dtype, device=device)
            pre_nms_joints = torch.zeros((batch_size, num_pre_nms_predictions, 17, 3), dtype=dtype, device=device)
            selected_indexes = torch.zeros((max_predictions_per_image // 2, 3), dtype=torch.int64, device=device)

            torch.onnx.export(
                PoseNMSAndReturnAsFlatResult(
                    batch_size=batch_size, num_pre_nms_predictions=num_pre_nms_predictions, max_predictions_per_image=max_predictions_per_image
                ),
                args=(pre_nms_boxes, pre_nms_scores, pre_nms_joints, selected_indexes),
                f=onnx_file,
                input_names=["input_pre_nms_boxes", "input_pre_nms_scores", "input_pre_nms_joints", "selected_indexes"],
                output_names=["flat_predictions"],
                dynamic_axes={
                    "input_pre_nms_boxes": {},
                    "input_pre_nms_scores": {},
                    "input_pre_nms_joints": {2: "num_joints"},
                    "selected_indexes": {0: "num_predictions"},
                    "flat_predictions": {0: "num_predictions"},
                },
            )

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            return convert_format_graph


def attach_onnx_pose_nms(
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
    Attach ONNX NMS stage to the pose estimation predictions.
    The model should have exactly two outputs: pred_boxes and pred_scores.
        - pred_boxes: [batch_size, num_pre_nms_predictions, 4]
        - pred_scores: [batch_size, num_pre_nms_predictions, 1]
        - pred_joints: [batch_size, num_pre_nms_predictions, num_joints, 3]
    This function will add the NMS layer to the model and return predictions in the format defined by output_format.

    :param onnx_model_path:           Input ONNX model path
    :param output_onnx_model_path:    Output ONNX model path. Can be the same as input model path.
    :param num_pre_nms_predictions:
    :param batch_size:                The batch size used for the inference.
    :param max_predictions_per_image: Maximum number of predictions per image
    :param confidence_threshold:      The confidence threshold to use for detections.
    :param nms_threshold:             The NMS threshold to use for detections.
    :param output_predictions_format: The output format of the predictions. Can be "flat" or "batch".

    If output_format equals to "flat":
    A single tensor of [N, K] will be returned, where N is the total number of detections across all images in the batch
    and K is computed as 6 + (num_joints) * 3
    Each row will contain [image_index, x1, y1, x2, y2, confidence, joint0_x, joint0_y, joint0_conf, joint1_x, joint1_y, joint1_conf, ...].

    If output_format equals to "batch" format:
    A tuple of 4 tensors (num_detections, detection_boxes, detection_scores, detection_poses) will be returned:
    - A tensor of [batch_size, 1] containing the image indices for each detection.
    - A tensor of [batch_size, max_output_boxes, 4] containing the bounding box coordinates for each detection in [x1, y1, x2, y2] format.
    - A tensor of [batch_size, max_output_boxes] containing the confidence scores for each detection.
    - A tensor of [batch_size, max_output_boxes, num_joints, 3] containing the predicted pose coordinates and confidence scores for each joint.

    :param device:                    The device to use for the conversion.
    :return:                          Function returns None, instead it saves model with attached NMS to output_onnx_model_path
    """
    graph = gs.import_onnx(onnx.load(onnx_model_path))
    graph.fold_constants()

    pred_boxes, pred_scores, pred_joints = graph.outputs

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
        pred_joints_f32 = gs.Variable(
            name="pred_joints_f32",
            dtype=np.float32,
            shape=pred_joints.shape,
        )
        graph.layer(op="Cast", name="cast_boxes_to_fp32", inputs=[pred_boxes], outputs=[pred_boxes_f32], attrs={"to": TensorProto.FLOAT})
        graph.layer(op="Cast", name="cast_scores_to_fp32", inputs=[pred_scores], outputs=[pred_scores_f32], attrs={"to": TensorProto.FLOAT})
        graph.layer(op="Cast", name="cast_joints_to_fp32", inputs=[pred_joints], outputs=[pred_joints_f32], attrs={"to": TensorProto.FLOAT})

        pred_scores = pred_scores_f32
        pred_boxes = pred_boxes_f32
        pred_joints = pred_joints_f32
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

    graph.outputs = [pred_boxes, pred_scores, pred_joints, output_selected_indices]

    if output_predictions_format == DetectionOutputFormatMode.BATCH_FORMAT:
        convert_format_graph = PoseNMSAndReturnAsBatchedResult.as_graph(
            batch_size=batch_size,
            num_pre_nms_predictions=num_pre_nms_predictions,
            max_predictions_per_image=max_predictions_per_image,
            dtype=numpy_dtype_to_torch_dtype(np.float32),
            device=device,
        )
        graph = append_graphs(graph, convert_format_graph)
    elif output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
        convert_format_graph = PoseNMSAndReturnAsFlatResult.as_graph(
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

    model = gs.export_onnx(graph)
    onnx.shape_inference.infer_shapes(model)
    onnx.save(model, output_onnx_model_path)
    logger.debug(f"Saved ONNX model to {output_onnx_model_path}")
