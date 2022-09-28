# How to use your own objects in SuperGradients recipes ?

## 1. Introduction

To train a model, it is necessary to configure 4 main components. These components are aggregated into a single "main" recipe .yaml
file that inherits the aforementioned dataset, architecture, training and checkpoint params.

Recipes support out of the box every model, metric or loss that is implemented in SuperGradients, but you can easily extend this to any custom object that you need by "registering it". 

*Notes*:
 - *If you are not familiar with recipes, please check our [introduction to recipes notebook](https://colab.research.google.com/drive/15hHgRtryIRkyoDO6rdiK5UcA4Ec3MleV?usp=sharing).*
 - *All recipes can be found [here](https://github.com/Deci-AI/super-gradients/blob/master/docs/assets/SG_img/Training_Recipes.md)*



## 2. General flow
**In your python script**
1. Define your custom object of type:
   * metric: `torchmetrics.Metric`
   * model: `torch.nn.Module`
   * loss:  `torch.nn.modules.loss._Loss`
2. Import the associated register decorator:
   * metric: `from super_gradients.training.utils.registry import register_metric`
   * model: `from super_gradients.training.utils.registry import register_model`
   * loss: coming soon
3. Apply it on your object.
   * The decorator takes an optional `name: str` argument. If not specified, the decorated class name will be registered.

**In your recipe (.yaml)**
1. Define your recipe like in any other case (you can find examples [here](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes)).
2. Modify the recipe by using the registered name (see the following examples).


## 3. Examples
### A. Metric

*main.py*

```python
import omegaconf
import hydra

import torch
import torchmetrics

from super_gradients import Trainer, init_trainer
from super_gradients.common.registry.registry import register_metric


@register_metric('custom_accuracy')  # Will be registered as "custom_accuracy"
class CustomAccuracy(torchmetrics.Accuracy):
   def update(self, preds: torch.Tensor, target: torch.Tensor):
      if target.shape == preds.shape:
         target = target.argmax(1)  # Supports smooth labels
         super().update(preds=preds.argmax(1), target=target)


@hydra.main(config_path="recipes")
def main(cfg: omegaconf.DictConfig) -> None:
   Trainer.train_from_config(cfg)


init_trainer()
main()
```

*recipes/training_hyperparams/my_training_hyperparams.yaml* 
```yaml
... # Other training hyperparams

train_metrics_list:
  - custom_accuracy

valid_metrics_list:
  - custom_accuracy
```

*Launch the script*
```bash
python main.py --config-name=my_recipe.yaml
```


### B. Model
Coming soon

### C. Loss

*main.py*

```python
import omegaconf
import hydra

import torch

from super_gradients import Trainer, init_trainer
from super_gradients.common.registry.registry import register_loss


@register_loss("custom_rsquared_loss")
class CustomRSquaredLoss(torch.nn.modules.loss._Loss): # The Loss needs to inherit from torch _Loss class.
   def forward(self, output, target):
       criterion_mse = torch.nn.MSELoss()
       return 1 - criterion_mse(output, target).item() / torch.var(target).item()


@hydra.main(config_path="recipes")
def main(cfg: omegaconf.DictConfig) -> None:
   Trainer.train_from_config(cfg)


init_trainer()
main()
```

*recipes/training_hyperparams/my_training_hyperparams.yaml* 
```yaml
... # Other training hyperparams

loss: custom_rsquared_loss
```

*Launch the script*
```bash
python main.py --config-name=my_recipe.yaml
```

