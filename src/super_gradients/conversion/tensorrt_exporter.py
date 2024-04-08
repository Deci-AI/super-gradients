from super_gradients.common.registry.registry import register_exporter
from super_gradients.conversion.abstract_exporter import AbstractExporter


@register_exporter()
class TRTExporter(AbstractExporter):
    def __init__(self, output_path: str):
        self.output_path = output_path

    def export_fp16(self, model):
        pass

    def export_fp32(self, model):
        pass

    def export_quantized(self, original_model, quantized_result):
        pass
