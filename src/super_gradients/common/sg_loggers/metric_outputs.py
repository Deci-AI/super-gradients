from dataclasses import dataclass
from typing import Union, Dict

import numpy as np


@dataclass
class PlottableMetricOutput:
    """Metric output that can is represented as a graph.
    :attr image:    The image of the graph representing the metric value. This can be ROC or Precision/Recall curves for instance.
                    This image will be logged as an image.
    :attr scalar:   The scalar value of the metric that "summarizes" the graph (e.g. Area Under Curve). This value will be logged and displayed.
    """

    image: np.ndarray
    scalar: float


MetricOutput = Union[PlottableMetricOutput, float, int]


def get_scalar_metric_outputs(metric_outputs: Dict[str, MetricOutput]) -> Dict[str, Union[float, str]]:
    return {k: v.scalar if isinstance(v, PlottableMetricOutput) else v for k, v in metric_outputs.items()}


def get_plottable_metric_outputs(metric_outputs: Dict[str, MetricOutput]) -> Dict[str, PlottableMetricOutput]:
    return {k: v for k, v in metric_outputs.items() if isinstance(v, PlottableMetricOutput)}
