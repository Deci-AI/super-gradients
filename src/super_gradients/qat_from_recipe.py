"""
Example code for running QAT on SuperGradient's recipes.

General use: python -m super_gradients.qat_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

import hydra
from omegaconf import DictConfig

from super_gradients import init_trainer, Trainer
from super_gradients.module_interfaces import QuantizationResult


@hydra.main(config_path="recipes", version_base="1.2")
def quantize_from_config(cfg: DictConfig) -> None:
    result: QuantizationResult = Trainer.quantize_from_config(cfg)

    print("Quantized model saved to", result.exported_model_path)
    print("Validation result of quantized model:")
    common_metrics = set(result.original_metrics.keys()) & set(result.quantized_metrics.keys())
    longest_metric_name = max(len(metric) for metric in common_metrics)
    print(f"{str.rjust('Metric',longest_metric_name)} | Original | Quantized | Relative Change")
    for metric in common_metrics:
        quantized_value = result.quantized_metrics[metric]
        original_value = result.original_metrics[metric]
        relative_change = (quantized_value - original_value) / original_value
        print(f"{str.rjust(metric, longest_metric_name)} | {original_value:<8.4f} | {quantized_value:<9.4f} | {100 * relative_change:+.2f}%")


def main():
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    quantize_from_config()


if __name__ == "__main__":
    main()
