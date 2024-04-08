import abc
from typing import Union, Dict, Any
import dataclasses
from torch import nn
from torch.utils.data import DataLoader


@dataclasses.dataclass
class QuantizationResult:
    """
    :param original_model: The original model that came in to quantization function.
    :param quantized_model: The quantized model. The value may not be the same instance or have another class.
    :param metrics: The metrics of the quantized model computed on validation set.
    """

    original_model: nn.Module
    quantized_model: Union[nn.Module, Any]
    metrics: Dict[str, float]


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
        model,
        trainer,
        calibration_loader: DataLoader,
        validation_loader: DataLoader,
        validation_metrics,
    ) -> QuantizationResult:
        pass

    @abc.abstractmethod
    def export(self, original_model, quantization_result, exporter):
        pass
