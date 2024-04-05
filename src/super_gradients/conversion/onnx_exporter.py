from super_gradients.common.registry.registry import register_exporter
from super_gradients.conversion.abstract_exporter import AbstractExporter


@register_exporter()
class ONNXRuntimeExporter(AbstractExporter):
    pass
