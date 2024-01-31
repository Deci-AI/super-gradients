from dataclasses import dataclass
from typing import Union, Dict

import numpy
import numpy as np
import torch


@dataclass
class PlottableMetricOutput:
    """Metric output that can is represented as a graph.
    :attr image:    The image of the graph representing the metric value. This can be ROC or Precision/Recall curves for instance.
                    This image will be logged as an image, and will not be associated with a scalar value.
    """

    image: np.ndarray


ScalarOutput = Union[float, int, np.number, torch.Tensor]
MetricOutput = Union[PlottableMetricOutput, ScalarOutput]


def get_scalar_metric_outputs(metric_outputs: Dict[str, MetricOutput]) -> Dict[str, float]:
    """Only keep the metric outputs that are a scalar (as opposed to plots for instance).
    :param metric_outputs:  Dictionary of metric outputs.
    :return:                Dictionary of scalar metric outputs.
    """
    scalar_metric_outputs = {}
    for name, value in metric_outputs.items():
        if isinstance(value, (float, int)):
            scalar_metric_outputs[name] = float(value)
        elif isinstance(value, np.number):
            scalar_metric_outputs[name] = value.item()  # Convert numpy number to Python scalar
        elif isinstance(value, torch.Tensor):
            if value.ndim == 0:  # Check if it's a scalar tensor
                scalar_metric_outputs[name] = value.item()  # Convert scalar tensor to Python scalar
    return scalar_metric_outputs


def get_plottable_metric_outputs(metric_outputs: Dict[str, MetricOutput]) -> Dict[str, PlottableMetricOutput]:
    """Only keep the metric outputs that are a plottable (as opposed to scalars for instance).
    :param metric_outputs:  Dictionary of metric outputs.
    :return:                Dictionary of metrics that are plottable.
    """
    return {k: v for k, v in metric_outputs.items() if isinstance(v, PlottableMetricOutput)}