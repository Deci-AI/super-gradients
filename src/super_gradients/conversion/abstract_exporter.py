import abc


class AbstractExporter(abc.ABC):
    @abc.abstractmethod
    def export_from_onnx(self, source_onnx: str):
        pass
