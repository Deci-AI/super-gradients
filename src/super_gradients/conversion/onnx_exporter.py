import shutil

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_exporter
from super_gradients.conversion.abstract_exporter import AbstractExporter

logger = get_logger(__name__)


@register_exporter()
class ONNXExporter(AbstractExporter):
    def __init__(self, output_path: str):
        """
        :param output_path: Output path for the exported model. Currently only supports ONNX format.
        """
        self.output_path = output_path

    def export_from_onnx(self, source_onnx: str):
        shutil.copy(source_onnx, self.output_path)
        return self.output_path
