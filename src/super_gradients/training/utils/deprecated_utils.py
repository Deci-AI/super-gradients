from typing import Callable, Any

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def wrap_with_warning(cls: Callable, message: str) -> Any:
    """
    Emits a warning when target class of function is called.

    >>> from super_gradients.training.utils.deprecated_utils import wrap_with_warning
    >>> from super_gradients.training.utils.callbacks import EpochStepWarmupLRCallback, BatchStepLinearWarmupLRCallback
    >>>
    >>> LR_WARMUP_CLS_DICT = {
    >>>     "linear": wrap_with_warning(
    >>>         EpochStepWarmupLRCallback,
    >>>         message=f"Parameter `linear` has been made deprecated and will be removed in the next SG release. Please use `linear_epoch` instead",
    >>>     ),
    >>>     'linear_epoch`': EpochStepWarmupLRCallback,
    >>> }

    :param cls: A class or function to wrap
    :param message: A message to emit when this class is called
    :return: A factory method that returns wrapped class
    """

    def _inner_fn(*args, **kwargs):
        logger.warning(message)
        return cls(*args, **kwargs)

    return _inner_fn
