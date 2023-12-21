from typing import Optional, Union, Sequence, Tuple
import torch
from torch import Tensor
from torchmetrics import Metric, MeanSquaredError, MeanSquaredLogError, MeanAbsoluteError, MeanAbsolutePercentageError
from super_gradients.common.registry import register_metric
from super_gradients.common.object_names import Metrics


class DepthEstimationMetricBase(Metric):
    """
    Base class for depth estimation metrics, handling common processing steps.

    :param metric: The specific torchmetrics metric instance.
    :param ignore_val: Value to be ignored when computing metricsn. In depth estimation tasks, it is common
                      to have regions in the depth map where the ground truth depth is not available or unreliable (e.g.,
                      marked as -1 or a specific value). In such cases, setting `ignore_val` allows you to exclude these
                      regions from the metric computation. It is important that the dataset class providing the depth map
                      fills the corresponding regions of the image with this `ignore_val` value to ensure consistency in
                      metric calculations.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions.
    """

    def __init__(self, metric: Metric, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__()
        self.metric = metric
        self.ignore_val = ignore_val
        self.apply_sigmoid = apply_sigmoid

    def process_preds_and_target(self, preds: Union[Tensor, Sequence[Tensor]], target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Process predictions and target tensors for depth estimation metrics:
        - If a sequence is returned by the model -> sets preds to the first element
        - Squeezes the dummy dimension (i.e dim1) when preds.ndim == 4
        - Applies sigmoid to preds if apply_sigmoid is True
        - Removes entries to ignore where ignore_val is present in target

        :param preds: Model predictions, either a tensor or a sequence of tensors.
        :param target: Ground truth depth map.
        :return: Processed predictions and target tensors.
        """

        if isinstance(preds, Sequence):
            preds = preds[0]
        if self.apply_sigmoid:
            preds = torch.sigmoid(preds)
        if self.ignore_val is not None:
            non_ignored = preds != self.ignore_val
            preds = preds[non_ignored]
            target = target[non_ignored]
        return preds, target

    def update(self, preds: Tensor, target: Tensor):
        preds, target = self.process_preds_and_target(preds, target)
        self.metric.update(preds, target)

    def compute(self):
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()


@register_metric(Metrics.DepthMSE)
class DepthMSE(DepthEstimationMetricBase):
    """
    Mean Squared Error metric (squared) for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(metric=MeanSquaredError(squared=True), ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


@register_metric(Metrics.DepthRMSE)
class DepthRMSE(DepthEstimationMetricBase):
    """
    Root Mean Squared Error metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(metric=MeanSquaredError(squared=False), ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


@register_metric(Metrics.DepthMSLE)
class DepthMSLE(DepthEstimationMetricBase):
    """
    Mean Squared Logarithmic Error metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(metric=MeanSquaredLogError(), ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


@register_metric(Metrics.DepthMAE)
class DepthMAE(DepthEstimationMetricBase):
    """
    Mean Absolute Error (MAE) metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(metric=MeanAbsoluteError(), ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


@register_metric(Metrics.DepthMAPE)
class DepthMAPE(DepthEstimationMetricBase):
    """
    Mean Absolute Percentage Error (MAPE) metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(metric=MeanAbsolutePercentageError(), ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


@register_metric(Metrics.DELTAMETRIC)
class DeltaMetric(Metric):
    """
    Delta metric - returns the percentage of pixels s.t max(preds / target, target / preds) < delta

    Use inheritors for ignored values.

    :param: delta (float): Threshold value for delta metric.

    """

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta
        self.add_state("total_delta_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        self.total_pixels += target.numel()
        self.total_delta_pixels += self.compute_delta_pixels(preds, target)

    def compute_delta_pixels(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Compute delta metrics for depth estimation without support for ignored values.

        :param preds: Model predictions.
        :param target: Ground truth depth map.
        :return: Delta metric value.
        """
        ratio = torch.max(preds / target, target / preds)
        return torch.sum((ratio < self.delta).float())

    def compute(self):
        return self.total_delta_pixels / self.total_pixels


@register_metric(Metrics.DELTA1)
class Delta1(DepthEstimationMetricBase):
    """
    Delta1 metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(metric=DeltaMetric(delta=1.25), ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


@register_metric(Metrics.DELTA2)
class Delta2(DepthEstimationMetricBase):
    """
    Delta2 metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(metric=DeltaMetric(delta=1.25**2), ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


@register_metric(Metrics.DELTA3)
class Delta3(DepthEstimationMetricBase):
    """
    Delta3 metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(metric=DeltaMetric(delta=1.25**3), ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)
