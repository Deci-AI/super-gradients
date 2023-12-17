from typing import Tuple, Sequence, Union, Optional

import torch
from torch import Tensor
from torchmetrics import MeanSquaredError, MeanSquaredLogError, MeanAbsoluteError, MeanAbsolutePercentageError, Metric


class DepthEstimationMetricMixin:
    """
    A mixin class providing common functionality for depth estimation metrics.

    :param ignore_val: Value to be ignored when computing metrics.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
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

        if preds.ndim == 4:
            preds = preds.squeeze(1)

        if self.apply_sigmoid:
            preds = torch.sigmoid(preds)

        if self.ignore_val is not None:
            non_ignored = preds != self.ignore_val
            preds = preds[non_ignored]
            target = target[non_ignored]

        return preds, target


class MeanSquaredErrorWithIgnored(MeanSquaredError, DepthEstimationMetricMixin):
    """MeanAbsoluteError, MeanAbsolutePercentageError
    Mean Squared Error metric for depth estimation with support for ignored values.

    :param squared: If True returns MSE value, if False returns RMSE value.
    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, squared: bool = True, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(squared=squared)
        DepthEstimationMetricMixin.__init__(self, ignore_val, apply_sigmoid)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric with model predictions and target depth map.

        :param preds: Model predictions.
        :param target: Ground truth depth map.
        """
        preds, target = self.process_preds_and_target(preds, target)
        super().update(preds, target)


class MSE(MeanSquaredErrorWithIgnored):
    """
    Mean Squared Error metric (squared) for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(squared=True, ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


class RMSE(MeanSquaredErrorWithIgnored):
    """
    Root Mean Squared Error metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(squared=False, ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


class MSLE(MeanSquaredLogError, DepthEstimationMetricMixin):
    """
    Mean Squared Logarithmic Error metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__()
        DepthEstimationMetricMixin.__init__(self, ignore_val, apply_sigmoid)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric with model predictions and target depth map.

        :param preds: Model predictions.
        :param target: Ground truth depth map.
        """
        preds, target = self.process_preds_and_target(preds, target)
        super().update(preds, target)


class MAE(MeanAbsoluteError, DepthEstimationMetricMixin):
    """
    Mean Absolute Error (MAE) metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__()
        DepthEstimationMetricMixin.__init__(self, ignore_val, apply_sigmoid)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric with model predictions and target depth map.

        :param preds: Model predictions.
        :param target: Ground truth depth map.
        """
        preds, target = self.process_preds_and_target(preds, target)
        super().update(preds, target)


class MAPE(MeanAbsolutePercentageError, DepthEstimationMetricMixin):
    """
    Mean Absolute Percentage Error (MAPE) metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__()
        DepthEstimationMetricMixin.__init__(self, ignore_val, apply_sigmoid)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric with model predictions and target depth map.

        :param preds: Model predictions.
        :param target: Ground truth depth map.
        """
        preds, target = self.process_preds_and_target(preds, target)
        super().update(preds, target)


class DeltaMetric(Metric, DepthEstimationMetricMixin):
    """
    Delta metrics for depth estimation with support for ignored values.

    :param deltas: List of threshold values for delta metrics.
    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, delta: float, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        Metric.__init__(self)
        DepthEstimationMetricMixin.__init__(self, ignore_val, apply_sigmoid)
        self.delta = delta
        self.add_state("total_delta_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self.process_preds_and_target(preds, target)
        self.total_pixels += target.numel()
        self.total_delta_pixels += self.compute_delta_pixels(preds, target)

    def compute_delta_pixels(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Compute delta metrics for depth estimation with support for ignored values.

        :param preds: Model predictions.
        :param target: Ground truth depth map.
        :param delta: Threshold value for delta metrics.
        :return: Delta metric value.
        """
        ratio = torch.max(preds / target, target / preds)
        delta_metric = torch.mean((ratio < self.delta).float())
        return delta_metric

    def compute(self):
        return self.total_delta_pixels / self.total_pixels


class Delta1(DeltaMetric):
    """
    Delta1 metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(delta=1.25, ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


class Delta2(DeltaMetric):
    """
    Delta2 metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(delta=1.25**2, ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)


class Delta3(DeltaMetric):
    """
    Delta3 metric for depth estimation with support for ignored values.

    :param ignore_val: Value to be ignored when computing the metric.
    :param apply_sigmoid: Whether to apply the sigmoid function to predictions before updating the metric.
    """

    def __init__(self, ignore_val: Optional[float] = None, apply_sigmoid: bool = False):
        super().__init__(delta=1.25**3, ignore_val=ignore_val, apply_sigmoid=apply_sigmoid)
