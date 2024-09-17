import dataclasses
from typing import Union, Dict, Any

from torch import nn
from torch.utils.data import DataLoader

__all__ = ["QuantizationResult"]


@dataclasses.dataclass
class QuantizationResult:
    """
    :param original_model: The original model that came in to quantization function.
    :param quantized_model: The quantized model. The value may not be the same instance or have another class.
    :param original_metrics: The metrics of the original model computed on validation set.
    :param quantized_metrics: The metrics of the quantized model computed on validation set.
    :param export_path: The path to the exported model. If the model was not exported, the value is None.
    :param export_result: The result of the export operation. If the model was not exported, the value is None.
           Can be an instance of PoseEstimationModelExportResult, ObjectDetectionModelExportResult, SegmentationModelExportResult.
    """

    original_model: nn.Module
    original_metrics: Dict[str, float]

    quantized_model: Union[nn.Module, Any]
    quantized_metrics: Dict[str, float]

    calibration_dataloader: DataLoader

    export_path: Union[None, str]
    export_result: Union[None, Any] = None

    def summary_str(self) -> str:
        """
        Returns a summary string of the quantization result with is a table showing metrics of original and quantized models.
        :return:
        """

        common_metrics = set(self.original_metrics.keys()) & set(self.quantized_metrics.keys())
        longest_metric_name = max(len(metric) for metric in common_metrics)
        longest_metric_name = max(longest_metric_name, len("Metric"))
        summary = f"{str.rjust('Metric',longest_metric_name)} | Original | Quantized | Relative Change\n"
        for metric in common_metrics:
            original_value = float(self.original_metrics[metric])
            quantized_value = float(self.quantized_metrics[metric])
            relative_change = (quantized_value - original_value) / (original_value + 1e-6)
            summary += f"{str.rjust(metric, longest_metric_name)} | {original_value:<8.4f} | {quantized_value:<9.4f} | {100 * relative_change:+.2f}%\n"
        return summary

    def summary_df(self):
        """
        Returns a summary DataFrame of the quantization result with is a table showing metrics of original and quantized models.
        This method is preferred over `summary_str` when you want to show summary in Jupyter Notebooks as you can simply visualize
        the DataFrame using built-in Jupyter tools.

        This method requires pandas to be installed.
        :return:
        """
        import pandas as pd
        import collections

        df = collections.defaultdict(list)
        common_metrics = set(self.original_metrics.keys()) & set(self.quantized_metrics.keys())
        for metric in common_metrics:
            original_value = float(self.original_metrics[metric])
            quantized_value = float(self.quantized_metrics[metric])
            relative_change = (quantized_value - original_value) / (original_value + 1e-6)
            df["Metric"].append(metric)
            df["Original"].append(original_value)
            df["Quantized"].append(quantized_value)
            df["Relative Change (%)"].append(100 * relative_change)

        return pd.DataFrame.from_dict(df).sort_values(by="Metric")
