import abc
import dataclasses


class TimeUnit(abc.ABC):
    """
    Abstract class for time units. This is used to explicitly log the time unit of a metric/loss.
    """

    @abc.abstractmethod
    def get_value(self):
        ...

    @abc.abstractmethod
    def get_name(self):
        ...


@dataclasses.dataclass
class EpochNumber(TimeUnit):
    """
    A time unit for epoch number.
    """

    value: float

    def get_value(self):
        return self.value

    def get_name(self):
        return "epoch"


@dataclasses.dataclass
class GlobalBatchStepNumber(TimeUnit):
    """
    A time unit for representing total number of batches processed, including training and validation ones.
    Suppose training loader has 320 batches and validation loader has 80 batches.
    If the current epoch index is 2 (zero-based), and we are on validation loader and current index is 50 (zero-based),
    then the global batch step is (320 + 80) * 3 + 320 + 50 = 1570.
    """

    value: float

    def get_value(self):
        return self.value

    def get_name(self):
        return "global_batch_step"
