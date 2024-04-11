import abc

from super_gradients.module_interfaces import QuantizationResult


class AbstractQuantizer(abc.ABC):
    """
    An abstract class for model quantization.
    Depending on the implementation and target framework, the quantization process may vary.
    Therefore, the quantization interface is defined in a broad manner using a trainer and a config.
    """

    @abc.abstractmethod
    def quantize(
        self,
        cfg,
        model,
        trainer,
    ) -> QuantizationResult:
        """
        Quantizes a model using the provided trainer and config.
        :param cfg: A config is a full recipe for training a model with `quantization_params` section defined.
        :param model: An input model to be quantized
        :param trainer: An instance of Trainer class to use for validating and (optionaly) training the quantized model (for QAT).
        :return: An instance of QuantizationResult class containing the quantized model and additional information.
        """
        pass
