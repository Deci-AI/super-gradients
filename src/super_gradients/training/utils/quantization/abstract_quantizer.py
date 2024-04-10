import abc

from super_gradients.module_interfaces import QuantizationResult
from torch import nn
from torch.utils.data import DataLoader


class AbstractQuantizer(abc.ABC):
    """
    An abstract class for quantization.
    """

    @abc.abstractmethod
    def ptq(
        self,
        model: nn.Module,
        trainer,
        calibration_loader: DataLoader,
        validation_loader: DataLoader,
        validation_metrics,
    ) -> QuantizationResult:
        pass

    @abc.abstractmethod
    def qat(
        self,
        cfg,
        model,
        trainer,
    ) -> QuantizationResult:
        pass
