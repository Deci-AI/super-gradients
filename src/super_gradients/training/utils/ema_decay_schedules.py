import math
from abc import abstractmethod

__all__ = ["IDecayFunction", "ConstantDecay", "ThresholdDecay", "ExpDecay", "EMA_DECAY_FUNCTIONS"]


class IDecayFunction:
    """
    Interface for EMA decay schedule. The decay schedule is a function of the maximum decay value and training progress.
    Usually it gradually increase EMA from to the maximum value. The exact ramp-up schedule is defined by the concrete
    implementation.
    """

    @abstractmethod
    def __call__(self, decay: float, step: int, total_steps: int) -> float:
        """

        :param decay: The maximum decay value.
        :param step: Current training step. The unit-range training percentage can be obtained by `step / total_steps`.
        :param total_steps:  Total number of training steps.
        :return: Computed decay value for a given step.
        """
        pass


class ConstantDecay(IDecayFunction):
    """
    Constant decay schedule.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, decay: float, step: int, total_steps: int) -> float:
        return decay


class ThresholdDecay(IDecayFunction):
    """
    Gradually increase EMA decay from 0.1 to the maximum value using following formula: min(decay, (1 + step) / (10 + step))
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, decay: float, step, total_steps: int) -> float:
        return min(decay, (1 + step) / (10 + step))


class ExpDecay(IDecayFunction):
    """
    Gradually increase EMA decay from 0.1 to the maximum value using following formula: decay * (1 - math.exp(-x * self.beta))

    """

    def __init__(self, beta: float, **kwargs):
        self.beta = beta

    def __call__(self, decay: float, step, total_steps: int) -> float:
        x = step / total_steps
        return decay * (1 - math.exp(-x * self.beta))


EMA_DECAY_FUNCTIONS = {"constant": ConstantDecay, "threshold": ThresholdDecay, "exp": ExpDecay}
