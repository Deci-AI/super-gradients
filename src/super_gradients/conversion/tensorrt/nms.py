import os
import tempfile

import numpy as np
import onnx
import torch
from torch import nn, Tensor

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.conversion.conversion_enums import DetectionOutputFormatMode
from super_gradients.conversion.conversion_utils import numpy_dtype_to_torch_dtype
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_fail_with_instructions
from super_gradients.conversion.onnx.utils import append_graphs

logger = get_logger(__name__)

gs = import_onnx_graphsurgeon_or_fail_with_instructions()


class ConvertTRTFormatToFlatTensor(nn.Module):
    __constants__ = ["batch_size", "max_predictions_per_image"]

    def __init__(self, batch_size: int, max_predictions_per_image: int):
        super().__init__()
        self.batch_size = batch_size
        self.max_predictions_per_image = max_predictions_per_image

    def forward(self, num_predictions: Tensor, pred_boxes: Tensor, pred_scores: Tensor, pred_classes: Tensor) -> Tensor:
        """
        Convert the predictions from "batch" format to "flat" tensor.
        :param num_predictions: [B,1] The number of predictions for each image in the batch.
        :param pred_boxes: [B, max_predictions_per_image, 4] The predicted bounding boxes for each image in the batch.
        :param pred_scores: [B, max_predictions_per_image] The predicted scores for each image in the batch.
        :param pred_classes: [B, max_predictions_per_image] The predicted classes for each image in the batch.
        :return: Tensor of shape [N, 7] The predictions in flat tensor format.
            N is the total number of predictions in the entire batch.
            Each row will contain [image_index, x1, y1, x2, y2, class confidence, class index] values.

        """
        batch_indexes = (
            torch.arange(start=0, end=self.batch_size, step=1, device=num_predictions.device).view(-1, 1).repeat(1, pred_scores.shape[1])
        )  # [B, max_predictions_per_image]

        preds_indexes = (
            torch.arange(start=0, end=self.max_predictions_per_image, step=1, device=num_predictions.device).view(1, -1, 1).repeat(self.batch_size, 1, 1)
        )  # [B, max_predictions_per_image, 1]

        flat_predictions = torch.cat(
            [
                preds_indexes.to(dtype=pred_scores.dtype),
                batch_indexes.unsqueeze(-1).to(dtype=pred_scores.dtype),
                pred_boxes,
                pred_scores.unsqueeze(dim=-1),
                pred_classes.unsqueeze(dim=-1).to(pred_scores.dtype),
            ],
            dim=-1,
        )  # [B, max_predictions_per_image, 8]

        num_predictions = num_predictions.repeat(1, self.max_predictions_per_image)  # [B, max_predictions_per_image]

        mask = (flat_predictions[:, :, 0] < num_predictions) & (flat_predictions[:, :, 1] == batch_indexes)  # [B, max_predictions_per_image]

        flat_predictions = flat_predictions[mask]  # [N, 7]
        return flat_predictions[:, 1:]

    @classmethod
    def as_graph(cls, batch_size: int, max_predictions_per_image: int, dtype: torch.dtype, device: torch.device) -> gs.Graph:
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_file = os.path.join(tmpdirname, "ConvertTRTFormatToFlatTensor.onnx")

            num_detections = torch.randint(1, max_predictions_per_image, (batch_size, 1), dtype=torch.int32, device=device)
            pred_boxes = torch.zeros((batch_size, max_predictions_per_image, 4), dtype=dtype, device=device)
            pred_scores = torch.zeros((batch_size, max_predictions_per_image), dtype=dtype, device=device)
            pred_classes = torch.zeros((batch_size, max_predictions_per_image), dtype=torch.int32, device=device)

            torch.onnx.export(
                ConvertTRTFormatToFlatTensor(batch_size=batch_size, max_predictions_per_image=max_predictions_per_image).to(device=device, dtype=dtype),
                args=(num_detections, pred_boxes, pred_scores, pred_classes),
                f=onnx_file,
                input_names=["num_detections", "pred_boxes", "pred_scores", "pred_classes"],
                output_names=["flat_predictions"],
                dynamic_axes={"flat_predictions": {0: "num_predictions"}},
            )

            convert_format_graph = gs.import_onnx(onnx.load(onnx_file))
            return convert_format_graph


def attach_tensorrt_nms(
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
    Attach TensorRT NMS plugin to the ONNX model

    :param onnx_model_path:
    :param output_onnx_model_path:
    :param max_predictions_per_image: Maximum number of predictions per image
    :param precision:
    :param batch_size:
    :return:
    """
    graph = gs.import_onnx(onnx.load(onnx_model_path))
    # graph.fold_constants()

    # Do shape inference
    # iteratively_infer_shapes(graph)

    pred_boxes, pred_scores = graph.outputs
    logger.debug(f"op_inputs: {pred_boxes}, {pred_scores}")
    op = "EfficientNMS_TRT"
    attrs = {
        "plugin_version": "1",
        "background_class": -1,  # no background class
        "max_output_boxes": max_predictions_per_image,
        "score_threshold": confidence_threshold,
        "iou_threshold": nms_threshold,
        "score_activation": False,
        "box_coding": 0,
    }

    # NMS Outputs
    output_num_detections = gs.Variable(
        name="num_dets",
        dtype=np.int32,
        shape=[batch_size, 1],
    )  # A scalar indicating the number of valid detections per batch image.
    output_boxes = gs.Variable(
        name="det_boxes",
        dtype=pred_boxes.dtype,
        shape=[batch_size, max_predictions_per_image, 4],
    )
    output_scores = gs.Variable(
        name="det_scores",
        dtype=pred_scores.dtype,
        shape=[batch_size, max_predictions_per_image],
    )
    output_labels = gs.Variable(
        name="det_classes",
        dtype=np.int32,
        shape=[batch_size, max_predictions_per_image],
    )

    op_outputs = [output_num_detections, output_boxes, output_scores, output_labels]

    # Create the NMS Plugin node with the selected inputs. The outputs of the node will also
    # become the final outputs of the graph.
    graph.layer(op=op, name="batched_nms", inputs=[pred_boxes, pred_scores], outputs=op_outputs, attrs=attrs)
    logger.info(f"Created NMS plugin '{op}' with attributes: {attrs}")

    graph.outputs = op_outputs

    if output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
        convert_format_graph = ConvertTRTFormatToFlatTensor.as_graph(
            batch_size=batch_size, max_predictions_per_image=max_predictions_per_image, dtype=numpy_dtype_to_torch_dtype(pred_boxes.dtype), device=device
        )
        graph = append_graphs(graph, convert_format_graph)
    elif output_predictions_format == DetectionOutputFormatMode.BATCH_FORMAT:
        pass
    else:
        raise NotImplementedError(f"Currently not supports output_predictions_format: {output_predictions_format}")

    # Final cleanup & save
    graph.cleanup().toposort()
    # iteratively_infer_shapes(graph)

    logger.debug(f"Final graph outputs: {graph.outputs}")

    model = gs.export_onnx(graph)
    onnx.save(model, output_onnx_model_path)
    logger.debug(f"Saved ONNX model to {output_onnx_model_path}")
