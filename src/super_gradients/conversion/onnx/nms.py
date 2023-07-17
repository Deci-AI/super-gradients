import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def iteratively_infer_shapes(graph):
    """
    Sanitize the graph by cleaning any unconnected nodes, do a topological resort,
    and fold constant inputs values. When possible, run shape inference on the
    ONNX graph to determine tensor shapes.
    """
    logger.debug("Performing shape inference & folding.")
    for _ in range(3):
        count_before = len(graph.nodes)

        graph.cleanup().toposort()
        try:
            for node in graph.nodes:
                for o in node.outputs:
                    o.shape = None
            model = gs.export_onnx(graph)
            model = shape_inference.infer_shapes(model)
            graph = gs.import_onnx(model)
        except Exception as e:
            logger.debug(f"Shape inference could not be performed at this time:\n{e}")
        try:
            graph.fold_constants(fold_shapes=True)
        except TypeError as e:
            logger.error("This version of ONNX GraphSurgeon does not support folding shapes, " f"please upgrade your onnx_graphsurgeon module. Error:\n{e}")
            raise

        count_after = len(graph.nodes)
        if count_before == count_after:
            # No new folding occurred in this iteration, so we can stop for now.
            break
        logger.debug(f"Folded {count_before - count_after} constants.")


def attach_onnx_nms(
    onnx_model_path: str,
    output_onnx_model_path,
    detections_per_img: int,
    confidence_threshold: float,
    nms_threshold: float,
    precision: str = "fp32",
    batch_size: int = 1,
):
    """
    Attach ONNX NMS plugin to the ONNX model

    :param onnx_model_path:
    :param output_onnx_model_path:
    :param precision:
    :param batch_size:
    :return:
    """
    graph = gs.import_onnx(onnx.load(onnx_model_path))
    graph.fold_constants()

    # Do shape inference
    iteratively_infer_shapes(graph)

    op_inputs = graph.outputs
    logger.debug(f"op_inputs: {op_inputs}")
    op = "NonMaxSuppression"
    attrs = {
        "center_point_box": 0,
    }

    if precision == "fp32":
        dtype_output = np.float32
    elif precision == "fp16":
        dtype_output = np.float16
    else:
        raise NotImplementedError(f"Currently not supports precision: {precision}")

    # NMS Inputs
    # input_max_output_boxes_per_class = gs.Constant("max_output_boxes_per_class", max_output_boxes_per_class)
    # input_iou_threshold = gs.Constant("nms_threshold", nms_threshold)
    # input_score_threshold = gs.Constant("score_threshold", confidence_threshold)

    # NMS Outputs
    output_num_detections = gs.Variable(
        name="num_dets",
        dtype=np.int32,
        shape=[batch_size, 1],
    )  # A scalar indicating the number of valid detections per batch image.
    output_boxes = gs.Variable(
        name="det_boxes",
        dtype=dtype_output,
        shape=[batch_size, detections_per_img, 4],
    )
    output_scores = gs.Variable(
        name="det_scores",
        dtype=dtype_output,
        shape=[batch_size, detections_per_img],
    )
    output_labels = gs.Variable(
        name="det_classes",
        dtype=np.int32,
        shape=[batch_size, detections_per_img],
    )

    op_outputs = [output_num_detections, output_boxes, output_scores, output_labels]

    # Create the NMS Plugin node with the selected inputs. The outputs of the node will also
    # become the final outputs of the graph.
    graph.layer(op=op, name="batched_nms", inputs=op_inputs, outputs=op_outputs, attrs=attrs)
    logger.info(f"Created NMS plugin '{op}' with attributes: {attrs}")

    graph.outputs = op_outputs

    iteratively_infer_shapes(graph)

    # Final cleanup & save
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx.save(model, output_onnx_model_path)
    logger.debug(f"Saved ONNX model to {output_onnx_model_path}")
