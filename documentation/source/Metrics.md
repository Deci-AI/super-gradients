# Metrics in SG

The purpose of metrics in general is to allow you to monitor and quantify the training process, therefore they are an essential component in every deep learning training process.
For this purpose, we leverage the [torchmetrics](https://torchmetrics.rtfd.io/en/latest/) library.
From `torchmetrics` homepage:

    "TorchMetrics is a collection of 90+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:
    
    - A standardized interface to increase reproducibility
    
    - Reduces Boilerplate
    
    - Distributed-training compatible
    
    - Rigorously tested
    
    - Automatic accumulation over batches
    
    - Automatic synchronization between multiple devices"

SG is compatible with any module metric implemented by torchmetrics (see full list [here](https://torchmetrics.rtfd.io/en/latest/)).
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

## Basic Usage of Implemented Metrics in SG:

For coded training scripts (i.e not [using configuration files](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/configuration_files.md)), the most basic usage is simply passing the metric objects through
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
Also notice the `metric_to_watch` training params that is set to `Accuracy` and `greater_metric_to_watch_is_better=True` meaning that we will monitor the validation accuracy and save checkpoints according to it.
Open any of the [tutorial notebooks](https://github.com/Deci-AI/super-gradients#getting-started) to see the metrics monitoring in action.
For more info on checkpoints and logs follow our SG checkpoints tutorial.

Equivalently, for [training with configuration files](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/configuration_files.md), your `my_training_hyperparams.yaml` would contain:
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
