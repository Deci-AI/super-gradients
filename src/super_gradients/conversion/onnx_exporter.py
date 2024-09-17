import shutil

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_exporter
from super_gradients.conversion.abstract_exporter import AbstractExporter

logger = get_logger(__name__)


@register_exporter()
class ONNXExporter(AbstractExporter):
    def __init__(self):
        pass

    def export_from_onnx(self, source_onnx: str, output_file: str) -> str:
        if source_onnx != output_file:
            shutil.copy(source_onnx, output_file)
        return output_file
