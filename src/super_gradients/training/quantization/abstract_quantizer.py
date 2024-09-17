import abc
from typing import Optional, Mapping, List

from super_gradients.module_interfaces import QuantizationResult
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics


class AbstractQuantizer(abc.ABC):
    """
    An abstract class for model quantization.
    Depending on the implementation and target framework, the quantization process may vary.
    Therefore, the quantization interface is defined in a broad manner using a trainer and a config.
    """

    @abc.abstractmethod
    def quantize_from_config(
        self,
        cfg,
        model,
        trainer,
    ) -> QuantizationResult:
        """
        Quantize a model using the provided trainer and config.
        :param cfg: A config is a full recipe for training a model with `quantization_params` section defined.
        :param model: An input model to be quantized
        :param trainer: An instance of Trainer class to use for validating and (optionally) training the quantized model (for QAT).
        :return: An instance of QuantizationResult class containing the quantized model and additional information.
        """
        pass

    def quantize_explicit(
        self,
        model: nn.Module,
        trainer,
        training_hyperparams: Optional[Mapping],
        train_loader: Optional[DataLoader],
        validation_loader: DataLoader,
        validation_metrics: List[torchmetrics.Metric],
        calibration_loader: DataLoader,
    ):
        """
        Quantize a model using the provided model, dataloaders and training_hyperparams.
        This function is helpful when you have pre-instantiated dataloaders / custom training hyperparameters or
        launching the quantization process not from config file but from the code.

        :param model: An input model to be quantized
        :param trainer: An instance of Trainer class to use for validating and (optionally) training the quantized model (for QAT).
        :param training_hyperparams:
        :param train_loader:
        :param validation_loader:
        :param validation_metrics:
        :param calibration_loader:
        :return: An instance of QuantizationResult class containing the quantized model and additional information.
        """
        pass
