import abc


class AbstractExporter(abc.ABC):
    @abc.abstractmethod
    def export_fp32(self, model):
        pass

    @abc.abstractmethod
    def export_fp16(self, model):
        pass

    @abc.abstractmethod
    def export_int8(self, model, calibration_dataset):
        pass
