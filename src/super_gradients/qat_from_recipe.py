"""
Example code for running QAT on SuperGradient's recipes.

General use: python -m super_gradients.qat_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""

from pprint import pprint

import hydra
from omegaconf import DictConfig

from super_gradients import init_trainer, Trainer


@hydra.main(config_path="recipes", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    result = Trainer.quantize_from_config(cfg)

    print("Validation result of quantized model:")
    pprint(result.valid_metrics_dict)

    if result.output_onnx_path is not None:
        print(f"ONNX model exported to {result.output_onnx_path}")

    if result.export_result is not None:
        print(result.export_result)


def main():
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()


if __name__ == "__main__":
    main()
