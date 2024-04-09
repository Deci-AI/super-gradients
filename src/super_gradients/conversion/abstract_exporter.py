import abc

from super_gradients.conversion import ExportParams
from torch import nn


class AbstractExporter(abc.ABC):
    @abc.abstractmethod
    def export(self, model: nn.Module, export_params: ExportParams):
        pass

    @abc.abstractmethod
    def export_quantized(self, original_model: nn.Module, quantization_result, export_params: ExportParams):
        pass
