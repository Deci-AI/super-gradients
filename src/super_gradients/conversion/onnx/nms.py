import os
import tempfile
from typing import Tuple, Optional, Mapping

import numpy as np
import onnx
import onnx.shape_inference
import onnxsim
import torch
from onnx import TensorProto
from torch import nn, Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.conversion.conversion_enums import DetectionOutputFormatMode
from super_gradients.conversion.conversion_utils import numpy_dtype_to_torch_dtype
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_fail_with_instructions
from super_gradients.conversion.onnx.utils import append_graphs, iteratively_infer_shapes

logger = get_logger(__name__)

gs = import_onnx_graphsurgeon_or_fail_with_instructions()


class PickNMSPredictionsAndReturnAsBatchedResult(nn.Module):
    __constants__ = ("batch_size", "max_predictions_per_image")

    def __init__(self, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int):
        """
        Select the predictions from ONNX NMS node and return them in batch format.

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
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, pred_boxes: Tensor, pred_scores: Tensor, selected_indexes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Select the predictions that are output by the NMS plugin.
        :param pred_boxes:       [B, N, 4] tensor, float32 in XYXY format
        :param pred_scores:      [B, N, C] tensor, float32
        :param selected_indexes: [num_selected_indices, 3], int64 - each row is [batch_indexes, label_indexes, boxes_indexes]
        :return:                 A tuple of 4 tensors (num_detections, detection_boxes, detection_scores, detection_classes) will be returned:
                                 - A tensor of [batch_size, 1] containing the image indices for each detection.
                                 - A tensor of [batch_size, max_predictions_per_image, 4] containing the bounding box coordinates
                                   for each detection in [x1, y1, x2, y2] format.
                                 - A tensor of [batch_size, max_predictions_per_image] containing the confidence scores for each detection.
                                 - A tensor of [batch_size, max_predictions_per_image] containing the class indices for each detection.

        """
        batch_indexes = selected_indexes[:, 0]
        label_indexes = selected_indexes[:, 1]
        boxes_indexes = selected_indexes[:, 2]

        selected_boxes = pred_boxes[batch_indexes, boxes_indexes]
        selected_scores = pred_scores[batch_indexes, boxes_indexes, label_indexes]

        if self.batch_size == 1:
            pred_boxes = selected_boxes[: self.max_predictions_per_image]
            pred_scores = selected_scores[: self.max_predictions_per_image]
            pred_classes = label_indexes[: self.max_predictions_per_image].long()
            num_predictions = pred_boxes.size(0).reshape(1, 1)

            pad_size = self.max_predictions_per_image - pred_boxes.size(0)
            pred_boxes = torch.nn.functional.pad(pred_boxes, [0, 0, 0, pad_size], value=-1, mode="constant")
            pred_scores = torch.nn.functional.pad(pred_scores, [0, pad_size], value=-1, mode="constant")
            pred_classes = torch.nn.functional.pad(pred_classes, [0, pad_size], value=-1, mode="constant")

            return num_predictions, pred_boxes.unsqueeze(0), pred_scores.unsqueeze(0), pred_classes.unsqueeze(0)
        else:
            predictions = torch.cat([selected_boxes, selected_scores.unsqueeze(1), label_indexes.unsqueeze(1)], dim=1)

            batch_predictions = torch.zeros((self.batch_size, self.max_predictions_per_image, 6), dtype=predictions.dtype, device=predictions.device)

            image_indexes = torch.arange(start=0, end=self.batch_size, step=1, device=predictions.device)
            masks = image_indexes.view(self.batch_size, 1) == batch_indexes.view(1, selected_indexes.size(0))  # [B, L]

            # Add dummy row to mask and predictions to ensure that we always have at least one prediction per image
            # ONNX/TRT deals poorly with tensors that has zero dims, and we need to ensure that we always have at least one prediction per image
            masks = torch.cat([masks, torch.zeros((self.batch_size, 1), dtype=masks.dtype, device=predictions.device)], dim=1)  # [B, L+1]
            predictions = torch.cat([predictions, torch.zeros((1, 6), dtype=predictions.dtype, device=predictions.device)], dim=0)  # [L+1, 6]

            num_predictions = torch.sum(masks, dim=1, keepdim=True).long()
            num_predictions_capped = torch.clamp_max(num_predictions, self.max_predictions_per_image)

            for i in range(self.batch_size):
                selected_predictions = predictions[masks[i]]
                pad_size = self.num_pre_nms_predictions - selected_predictions.size(0)
                selected_predictions = torch.nn.functional.pad(selected_predictions, [0, 0, 0, pad_size], value=-1, mode="constant")
                selected_predictions = selected_predictions[0 : self.max_predictions_per_image]

                batch_predictions[i] = selected_predictions

            pred_boxes = batch_predictions[:, :, 0:4]
            pred_scores = batch_predictions[:, :, 4]
            pred_classes = batch_predictions[:, :, 5].long()

            return num_predictions_capped, pred_boxes, pred_scores, pred_classes

    @classmethod
    def as_graph(
        cls,
        batch_size: int,
        num_pre_nms_predictions: int,
        max_predictions_per_image: int,
        dtype: torch.dtype,
        device: torch.device,
        onnx_export_kwargs: Optional[Mapping] = None,
    ) -> gs.Graph:
        if onnx_export_kwargs is None:
            onnx_export_kwargs = {}
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
                **onnx_export_kwargs,
            )

            model_opt, check_ok = onnxsim.simplify(onnx_file)
            if not check_ok:
                raise RuntimeError(f"Failed to simplify ONNX model {onnx_file}")
            onnx.save(model_opt, onnx_file)

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            convert_format_graph = convert_format_graph.fold_constants().cleanup().toposort()
            convert_format_graph = iteratively_infer_shapes(convert_format_graph)
            return convert_format_graph


class PickNMSPredictionsAndReturnAsFlatResult(nn.Module):
    """
    Select the output from ONNX NMS node and return them in flat format.

    This module is NOT compatible with TensorRT engine (Tested on TensorRT 8.4.2, 8.5.3 and 8.6.1) when using batch size > 1.
    """

    __constants__ = ("batch_size", "num_pre_nms_predictions", "max_predictions_per_image")

    def __init__(self, batch_size: int, num_pre_nms_predictions: int, max_predictions_per_image: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, pred_boxes: Tensor, pred_scores: Tensor, selected_indexes: Tensor) -> Tensor:
        """
        Select the predictions that are output by the NMS plugin.
        :param pred_boxes:       [B, N, 4] tensor
        :param pred_scores:      [B, N, C] tensor
        :param selected_indexes: [num_selected_indices, 3] - each row is [batch_indexes, label_indexes, boxes_indexes]
                                 Indexes of predictions from same image (same batch_index) corresponds to sorted predictions (Confident first).
        :return:                 A single tensor of [Nout, 7] shape, where Nout is the total number of detections across all images in the batch.
                                 Each row will contain [image_index, x1, y1, x2, y2, class confidence, class index] values.
                                 Each image will have at most max_predictions_per_image detections.

        """
        batch_indexes = selected_indexes[:, 0]
        label_indexes = selected_indexes[:, 1]
        boxes_indexes = selected_indexes[:, 2]

        selected_boxes = pred_boxes[batch_indexes, boxes_indexes]
        selected_scores = pred_scores[batch_indexes, boxes_indexes, label_indexes]
        dtype = selected_scores.dtype

        flat_results = torch.cat(
            [batch_indexes.unsqueeze(-1).to(dtype), selected_boxes, selected_scores.unsqueeze(-1), label_indexes.unsqueeze(-1).to(dtype)], dim=1
        )  # [N, 7]

        if self.batch_size > 1:
            # Compute a mask of shape [N,B] where each row contains True if the corresponding prediction belongs to the corresponding batch index
            image_index = torch.arange(self.batch_size, dtype=batch_indexes.dtype, device=batch_indexes.device)

            detections_in_images_mask = image_index.view(1, self.batch_size) == batch_indexes.view(-1, 1)  # [num_selected_indices, B]

            # Compute total number of detections per image
            num_detections_per_image = torch.sum(detections_in_images_mask, dim=0, keepdim=True)  # [1, B]

            # Cap the number of detections per image to max_predictions_per_image
            num_detections_per_image = torch.clamp_max(num_detections_per_image, self.max_predictions_per_image)  # [1, B]

            # Calculate cumulative count of selected predictions for each batch index
            # This will give us a tensor of shape [num_selected_indices, B] where the value at each position
            # represents the number of predictions for the corresponding batch index up to that position.
            cumulative_counts = detections_in_images_mask.float().cumsum(dim=0)  # [num_selected_indices, B]

            # Create a mask to ensure we only keep max_predictions_per_image detections per image
            mask = ((cumulative_counts <= num_detections_per_image) & detections_in_images_mask).any(dim=1, keepdim=False)  # [N]

            final_results = flat_results[mask > 0]
        else:
            final_results = flat_results[: self.max_predictions_per_image]

        return final_results

    @classmethod
    def as_graph(
        cls,
        batch_size: int,
        num_pre_nms_predictions: int,
        max_predictions_per_image: int,
        dtype: torch.dtype,
        device: torch.device,
        onnx_export_kwargs: Optional[Mapping] = None,
    ) -> gs.Graph:
        if onnx_export_kwargs is None:
            onnx_export_kwargs = {}
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
                    "flat_predictions": {0: "num_output_predictions"},
                },
                **onnx_export_kwargs,
            )

            model_opt, check_ok = onnxsim.simplify(onnx_file)
            if not check_ok:
                raise RuntimeError(f"Failed to simplify ONNX model {onnx_file}")
            onnx.save(model_opt, onnx_file)

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            convert_format_graph = convert_format_graph.fold_constants().cleanup().toposort()
            convert_format_graph = iteratively_infer_shapes(convert_format_graph)

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
    onnx_export_kwargs: Optional[Mapping] = None,
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
        gs.Constant(name="max_output_boxes_per_class", values=np.array([num_pre_nms_predictions], dtype=np.int64)),
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
            onnx_export_kwargs=onnx_export_kwargs,
        )
        graph = append_graphs(graph, convert_format_graph)
    elif output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
        convert_format_graph = PickNMSPredictionsAndReturnAsFlatResult.as_graph(
            batch_size=batch_size,
            num_pre_nms_predictions=num_pre_nms_predictions,
            max_predictions_per_image=max_predictions_per_image,
            dtype=numpy_dtype_to_torch_dtype(np.float32),
            device=device,
            onnx_export_kwargs=onnx_export_kwargs,
        )
        graph = append_graphs(graph, convert_format_graph)
    else:
        raise ValueError(f"Invalid output_predictions_format: {output_predictions_format}")

    # Final cleanup & save
    graph = graph.toposort().fold_constants().cleanup()
    graph = iteratively_infer_shapes(graph)

    model = gs.export_onnx(graph)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, output_onnx_model_path)
    logger.debug(f"Saved ONNX model to {output_onnx_model_path}")
