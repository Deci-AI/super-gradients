import shutil

import openvino as ov
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_exporter
from super_gradients.conversion.abstract_exporter import AbstractExporter

logger = get_logger(__name__)


@register_exporter()
class OpenVinoExporter(AbstractExporter):
    def __init__(self, output_path: str, compress_to_fp16: bool = False):
        if output_path.endswith(".xml") or output_path.endswith(".onnx"):
            self.output_path = output_path
        else:
            raise ValueError(f"Unsupported output format: {output_path}. Only .onnx or .xml extensions are supported.")
        self.output_path = output_path
        self.compress_to_fp16 = compress_to_fp16

    def export_from_onnx(self, source_onnx: str):
        if self.output_path.endswith(".xml"):
            ov_model = ov.convert_model(source_onnx)
            ov.save_model(ov_model, self.output_path, compress_to_fp16=False)
        elif self.output_path.endswith(".onnx"):
            shutil.copy(source_onnx, self.output_path)
        else:
            raise ValueError(f"Unsupported output format: {self.output_path}")

        return self.output_path
