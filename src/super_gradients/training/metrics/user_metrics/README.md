<div "center">
  <img src="docs/assets/SG_img/SG - Horizontal.png" width="600"/>
 <br/><br/>

## Introduction
This page demonstrates how you can register your own metrics, so that SuperGradients can access it with a name `str`, for
example, when training from a recipe config.

## Usage
**In your python script**
1. Define your PyTorch metric (`torchmetrics.Metric`) in the new module.
2. Import the `@register` decorator 
`from super_gradients.training.utils.registry import register_metric` and apply it to your model.
   * The decorator can be applied directly to the class or to a function returning the class.
   * The decorator takes an optional `name: str` argument. If not specified, the decorated class/function name will be registered.

**In your recipe (.yaml)**
1. Define your recipe like in any other case
2. Replace (or add) your new metric in `train_metrics_list` or in `valid_metrics_list`


## Example

*main.py*
```python
import omegaconf
import hydra

import torch
import torchmetrics

from super_gradients import Trainer, init_trainer
from super_gradients.training.utils.registry import register_metric


@register_metric('custom_accuracy')  # Will be registered as "custom_accuracy"
class CustomAccuracy(torchmetrics.Metric):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if target.shape == preds.shape:
            target = target.argmax(1)  # Supports smooth labels
            super().update(preds=preds.argmax(1), target=target)


@hydra.main(config_path="config_path")
def main(cfg: omegaconf.DictConfig) -> None:
    Trainer.train_from_config(cfg)


init_trainer()
main()
```

*config_path/my_recipe.yaml* 
```yaml
...
   
train_metrics_list:
  - custom_accuracy
```

*Launch the script*
```bash
python main.py --config-name=my_recipe.yaml
```