# Metrics

The purpose of metrics is to allow you to monitor and quantify the training process. Therefore, metrics are an essential component in every deep learning training process.
For this purpose, we leverage the [torchmetrics](https://torchmetrics.rtfd.io/en/latest/) library.
From the `torchmetrics` homepage:

    "TorchMetrics is a collection of 90+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:
    
    - A standardized interface to increase reproducibility
    
    - Reduces Boilerplate
    
    - Distributed-training compatible
    
    - Rigorously tested
    
    - Automatic accumulation over batches
    
    - Automatic synchronization between multiple devices."

SG is compatible with any module metric implemented by torchmetrics (see complete list [here](https://torchmetrics.rtfd.io/en/latest/)).
Apart from the native `torchmetrics` implementations, SG implements some metrics as `torchmetrics.Metric` objects as well:

    Accuracy
    Top5
    DetectionMetrics
    IoU
    PixelAccuracy
    BinaryIOU
    Dice
    BinaryDice
    DetectionMetrics_050
    DetectionMetrics_075
    DetectionMetrics_050_095

## Basic Usage of Implemented Metrics

For coded training scripts (i.e., not [using configuration files](configuration_files.md)), the most basic usage is simply passing the metric objects through
`train_metrics_list` and `valid_metrics_list`:

```python
from super_gradients import Trainer
...
from super_gradients.training.metrics import Accuracy, Top5

trainer = Trainer("my_experiment")
train_dataloader = ...
valid_dataloader = ...
model = ...
train_params = {
    ...
    "train_metrics_list": [Accuracy(), Top5()],
    "valid_metrics_list": [Accuracy(), Top5()],
    "metric_to_watch": "Accuracy",
    "greater_metric_to_watch_is_better": True,
}
trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```

Now, the metrics progress over the training epochs (and validation) will be displayed and logged in the Tensorboards, and any 3rd party SG Logger (see integration with Weights & Biases and Clearml in [repo homepage](https://github.com/Deci-AI/super-gradients#-integration-to-weights-and-biases-)).
Metric results will be lowercase, with the appropriate suffix: `train_accuracy`, `train_top5`, `valid_accuracy`, `valid_top5`.
Also, notice the `metric_to_watch` set to `Accuracy` and `greater_metric_to_watch_is_better=True`, meaning that we will monitor the validation accuracy and save checkpoints according to it.
Open any of the [tutorial notebooks](https://github.com/Deci-AI/super-gradients#getting-started) to see the metrics monitoring in action.
For more info on checkpoints and logs, follow our SG checkpoints tutorial.

Equivalently, for [training with configuration files](configuration_files.md), your `my_training_hyperparams.yaml` would contain:
```yaml
defaults:
  - default_train_params
...
...
metric_to_watch: Accuracy
greater_metric_to_watch_is_better: True

train_metrics_list:                               # metrics for evaluation
  - Accuracy
  - Top5

valid_metrics_list:                               # metrics for evaluation
  - Accuracy
  - Top5
```

## Using Custom Metrics

Suppose you implemented your own `MyAccuracy` (more information on how to do so [here](https://torchmetrics.readthedocs.io/en/latest/pages/implement.html)), for coded training, you can pass an instance of it as done in the previous sub-section.
For [training with configuration files](configuration_files.md), first decorate your metric class with SG's `@register_metric` decorator:
```python
from torchmetrics import Metric
import torch
from super_gradients.common.registry import register_metric

@register_metric("my_accuracy")
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
```

Next, use the registered metric in your `my_training_hyperparams.yaml` by plugging in the registered name, just as if it was any other metric:
```yaml
defaults:
  - default_train_params
...
...
metric_to_watch: my_accuracy
greater_metric_to_watch_is_better: True

train_metrics_list:                               # metrics for evaluation
  - my_accuracy
...

valid_metrics_list:                               # metrics for evaluation
  - my_accuracy
...
```

Last, in your ``my_train_from_recipe_script.py`` file, import the newly registered class (even though the class itself is unused, just to trigger the registry):
        
```python
from omegaconf import DictConfig
import hydra
import pkg_resources
from my_accuracy import MyAccuracy
from super_gradients import Trainer, init_trainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
  Trainer.train_from_config(cfg)


def run():
  init_trainer()
  main()


if __name__ == "__main__":
  run()
```
