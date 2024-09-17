import abc


class AbstractExporter(abc.ABC):
    """
    An abstract class for exporting a model from ONNX representation to a specific framework.
    For instance, ONNX model can be exported to TFLite, OpenVINO or CoreML formats.
    This can be done by subclassing from this class and implementing the `export_from_onnx` method.
    """

    @abc.abstractmethod
    def export_from_onnx(self, source_onnx: str, output_file: str) -> str:
        """
        Exports a model from ONNX representation to an output file.
        A output filename extension and it's content should be determined by the subclass.
        :param source_onnx: Input ONNX model file path.
        :param output_file: Output file path of the exported model.
        :return: Output file path of the exported model.
        """
        pass
