import shutil

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_exporter
from super_gradients.conversion.abstract_exporter import AbstractExporter

logger = get_logger(__name__)


@register_exporter()
class TRTExporter(AbstractExporter):
    def __init__(self, output_path: str):
        """
        :param output_path: Output path for the exported model. Currently only supports ONNX format.
        """
        if not output_path.endswith(".onnx"):
            raise ValueError(f"Unsupported output format: {output_path}. Only .onnx extension is supported.")
        self.output_path = output_path

    def export_from_onnx(self, source_onnx: str):
        shutil.copy(source_onnx, self.output_path)
        raise ValueError(f"Unsupported output format: {self.output_path}")
