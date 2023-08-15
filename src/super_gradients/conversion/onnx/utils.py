from onnx import shape_inference

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_fail_with_instructions

logger = get_logger(__name__)

gs = import_onnx_graphsurgeon_or_fail_with_instructions()


def append_prefix_to_graph(graph: gs.Graph, prefix: str) -> gs.Graph:
    """
    Append a prefix to all nodes and outputs in the graph to avoid name collisions.
    :param graph: The graph to rename.
    :param prefix: The prefix to add to all nodes and outputs in the graph.
    :return: The modified new instance of graph.
    """
    for node in graph.nodes:
        node.name = prefix + node.name

    for output in graph.outputs:
        output.name = prefix + output.name

    return graph


def append_graphs(graph1: gs.Graph, graph2: gs.Graph, prefix: str = "graph2_") -> gs.Graph:
    """
    Append one graph to another. This function modify graph1 in place.
    Outputs from the first graph will be connected to inputs of the second graph.
    :param graph1: The first graph. Will be modified in place.
    :param graph2: The second graph to attach to the first graph.
    :param prefix: The prefix to add to all nodes and outputs in the second graph to avoid name collisions.
    :return: The first graph, with the second graph appended to it.
    """
    if len(graph1.outputs) != len(graph2.inputs):
        raise ValueError(f"Number of outputs ({len(graph1.outputs)}) does not match number of inputs ({len(graph2.inputs)})")

    merged_graph = graph1
    graph2 = append_prefix_to_graph(graph2, prefix)

    for node in graph2.nodes:
        merged_graph.nodes.append(node)

    # Actual stitching of the graphs
    for out, inp in zip(graph1.outputs, graph2.inputs):
        merged_graph.nodes.append(gs.Node(op="Identity", name=f"Identity_{out.name}_{inp.name}", inputs=[out], outputs=[inp]))

    merged_graph.outputs.clear()
    merged_graph.outputs = graph2.outputs

    merged_graph.toposort()
    # iteratively_infer_shapes(merged_graph)

    return merged_graph


def iteratively_infer_shapes(graph: gs.Graph) -> None:
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
            # for node in graph.nodes:
            #     for o in node.outputs:
            #         o.shape = None
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
