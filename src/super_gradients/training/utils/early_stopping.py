from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from typing import Optional
import torch
import numpy as np

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_callback
from super_gradients.common.object_names import Callbacks


logger = get_logger(__name__)


@register_callback(Callbacks.EARLY_STOP)
class EarlyStop(PhaseCallback):
    """
    Callback to monitor a metric and stop training when it stops improving.
    Inspired by pytorch_lightning.callbacks.early_stopping and tf.keras.callbacks.EarlyStopping
    """

    mode_dict = {"min": torch.lt, "max": torch.gt}
    supported_phases = (Phase.VALIDATION_EPOCH_END, Phase.TRAIN_EPOCH_END)

    def __init__(
        self,
        phase: Phase,
        monitor: str,
        mode: str = "min",
        min_delta: float = 0.0,
        patience: int = 3,
        check_finite: bool = True,
        threshold: Optional[float] = None,
        verbose: bool = False,
        strict: bool = True,
    ):
        """

        :param phase: Callback phase event.
        :param monitor: name of the metric to be monitored.
        :param mode: one of 'min', 'max'. In 'min' mode, training will stop when the quantity
           monitored has stopped decreasing and in 'max' mode it will stop when the quantity
           monitored has stopped increasing.
        :param min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
           change of less than `min_delta`, will count as no improvement.
        :param patience: number of checks with no improvement after which training will be stopped.
            One check happens after every phase event.
        :param check_finite: When set ``True``, stops training when the monitor becomes NaN or infinite.
        :param threshold: Stop training immediately once the monitored quantity reaches this threshold. For mode 'min'
            stops training when below threshold, For mode 'max' stops training when above threshold.
        :param verbose: If `True` print logs.
        :param strict: whether to crash the training if `monitor` is not found in the metrics.
        """
        super(EarlyStop, self).__init__(phase)

        if phase not in self.supported_phases:
            raise ValueError(f"EarlyStop doesn't support phase: {phase}, " f"excepted {', '.join([str(x) for x in self.supported_phases])}")
        self.phase = phase
        self.monitor_key = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.check_finite = check_finite
        self.threshold = threshold
        self.verbose = verbose
        self.strict = strict

        self.wait_count = 0

        if self.mode not in self.mode_dict:
            raise Exception(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")
        self.monitor_op = self.mode_dict[self.mode]
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1

        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    def _get_metric_value(self, metrics_dict):
        if self.monitor_key not in metrics_dict.keys():
            msg = f"Can't find EarlyStop monitor {self.monitor_key} in metrics_dict: {metrics_dict.keys()}"
            exception_cls = RuntimeError if self.strict else MissingMonitorKeyException
            raise exception_cls(msg)
        return metrics_dict[self.monitor_key]

    def _check_for_early_stop(self, current: torch.Tensor):
        should_stop = False

        # check if current value is Nan or inf
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor_key} = {current} is not finite." f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )

        # check if current value reached threshold value
        elif self.threshold is not None and self.monitor_op(current, self.threshold):
            should_stop = True
            reason = "Stopping threshold reached:" f" {self.monitor_key} = {current} {self.monitor_op} {self.threshold}." " Signaling Trainer to stop."

        # check if current is an improvement of monitor_key metric.
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            if torch.isfinite(self.best_score):
                reason = (
                    f"Metric {self.monitor_key} improved by {abs(self.best_score - current):.3f} >="
                    f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
                )
            else:
                reason = f"Metric {self.monitor_key} improved. New best score: {current:.3f}"
            self.best_score = current
            self.wait_count = 0

        # no improvement in monitor_key metric, check if wait_count is bigger than patience.
        else:
            self.wait_count += 1
            reason = f"Monitored metric {self.monitor_key} did not improve in the last {self.wait_count} records."
            if self.wait_count >= self.patience:
                should_stop = True
                reason += f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."

        return reason, should_stop

    def __call__(self, context: PhaseContext):
        try:
            current = self._get_metric_value(context.metrics_dict)
        except MissingMonitorKeyException as e:
            logger.warning(e)
            return

        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current)

        reason, self.should_stop = self._check_for_early_stop(current)

        # log reason message, and signal early stop if should_stop=True.
        if self.should_stop:
            self._signal_early_stop(context, reason)

        elif self.verbose:
            logger.info(reason)

    def _signal_early_stop(self, context: PhaseContext, reason: str):
        logger.info(reason)
        context.update_context(stop_training=True)


class MissingMonitorKeyException(Exception):
    """
    Exception raised for missing monitor key in metrics_dict.
    """

    pass
