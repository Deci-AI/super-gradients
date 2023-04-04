# Losses

SuperGradients can support any PyTorch-based loss function. Additionally, multiple Loss function implementations for various tasks are also supported:

    cross_entropy
    mse
    r_squared_loss
    shelfnet_ohem_loss
    shelfnet_se_loss
    yolox_loss
    yolox_fast_loss
    ssd_loss
    stdc_loss
    bce_dice_loss
    kd_loss
    dice_ce_edge_loss

All the above, are just string aliases for the underlying torch.nn.Module classes, implementing the specified loss functions.

## Basic Usage of Implemented Loss Functions

The most basic use case is when using a direct Trainer.train(...) call:

In your `my_training_script.py`:
```python
...
trainer = Trainer("external_criterion_test")
train_dataloader = ...
valid_dataloader = ...
model = ...

train_params = {
   ...
   "loss": "cross_entropy",
   "criterion_params": {}
   ...
}
trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```
Note that object names in SG are not case-sensitive nor symbol-sensitive, so `"CrossEntropy` could have been passed as well.
Since most IDEs support auto-completion, for your convenience, you can use our object_names module:
```python
from super_gradients.common.object_names import Losses
```
Then simply instead of "cross_entropy", use 
```python
Losses.CROSS_ENTROPY
```


Another use case is when using configuration files. For example, when training using train_from_recipe (or similar, when the underlying train method that is being called is Trainer.train_from_config(...)).

When doing so, in your `my_training_hyperparams.yaml` file:
```yaml
...

loss: yolox_loss

criterion_params:
   strides: [8, 16, 32]  # output strides of all yolo outputs
   num_classes: 80
```

Note that two `training_params` parameters define the loss function:  `loss` which defines the type of the loss, and`criterion_params` dictionary which will be unpacked to the underlying `yolox_loss` class constructor.

## Passing Instantiated nn.Module Objects as Loss Functions

SuperGradients also supports passing instantiated nn.Module Objects as demonstrated below:
When using a direct Trainer.train(...) call, in your `my_training_script.py` simply pass the instantiated nn.Module under the "loss" key inside training_params:
```python
...
trainer = Trainer("external_criterion_test")
train_dataloader = ...
valid_dataloader = ...
model = ...

train_params = {
    ...
    "loss": torch.nn.CrossEntropy()
    ...
}
trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)
```

Though not as convenient as using `register_loss` (discussed further into detail in the next sub-section), one can also equivalently instantiate objects when using train_from_recipe (or similar, when the underlying train method is Trainer.train_from_config(...) as demonstrated below:


In your `my_training_hyperparams.yaml` file:
```yaml
...

loss:
 _target_: torch.nn.CrossEntropy
```
  Note that when passing an instantiated loss object, `criterion_params` will be ignored.


## Using Your Own Loss

SuperGradients also supports user-defined loss functions assuming they are torch.nn.Module inheritors, and that their `forward` signature is in the form:

```python
import torch.nn

MyLoss(torch.nn.Module):
...
forward(preds, target):
...
```
And as the argument names suggest, the first argument is the model's output, and target is the label/ground truth (argument naming is arbitrary and does not need to be specifically 'preds' or 'target').
Loss functions accepting additional arguments in their `forward` method will be supported in the future.

### Using Your Own Loss - Logging Loss Outputs

In the most common case, where the loss function returns a single item for backprop the loss output will appear in
the logs, training logs (i.e Tensorboards and any other supported SGLogger, for more information on SGLoggers click [here](https://github.com/Deci-AI/super-gradients)), over epochs under <LOSS_CLASS.__name__>. 

forward(...) should return a (loss, loss_items) tuple where loss is the tensor used
for backprop (i.e what your original loss function returns), and loss_items should be a tensor of
shape (n_items) consisting of values computed during the forward pass which we desire to log over the
entire epoch. For example- the loss itself should always be logged. Another example is a scenario
where the computed loss is the sum of a few components we would like to log.

For example:
```python
class MyLoss(_Loss):
    ...
    def forward(self, inputs, targets):
        ...
        total_loss = comp1 + comp2
        loss_items = torch.cat((total_loss.unsqueeze(0),comp1.unsqueeze(0), comp2.unsqueeze(0)).detach()
        return total_loss, loss_items


train_params = {
     ...,
     "loss": MyLoss(),
     "metric_to_watch": "MyLoss/loss_0"
 }

Trainer.train(
    ...,
    train_params=train_params
)
```


The above snippet will log `MyLoss2/loss_0`, `MyLoss2/loss_1` and  `MyLoss2/loss_2` as they have been named by their positional index in loss_items.
Note we also defined "MyLoss2/loss_0" to be our watched metric which means we save our checkpoint every epoch we reach the best loss score.

For more visibility, you can also set a "component_names" property in the loss class,
to be a list of strings, of length n_items whose ith element is the name of the ith entry in loss_items.
Then each item will be logged, rendered on the tensorboard, and "watched" (i.e saving model checkpoints
according to it) under `<LOSS_CLASS.__name__>/<COMPONENT_NAME>`. 

For example:
```python
class MyLoss(_Loss):
    ...
    def forward(self, inputs, targets):
        ...
        total_loss = comp1 + comp2
        loss_items = torch.cat((total_loss.unsqueeze(0),comp1.unsqueeze(0), comp2.unsqueeze(0)).detach()
        return total_loss, loss_items
    ...
    @property
    def component_names(self):
        return ["total_loss", "my_1st_component", "my_2nd_component"]

train_params = {
     ...,
     "loss": MyLoss(),
     "metric_to_watch": "MyLoss/my_1st_component"
 }

Trainer.train(
    ...,
    train_params=train_params
)
```

The above code will log and monitor `MyLoss/total_loss`, `MyLoss/my_1st_component` and `MyLoss/my_2nd_component`.


Since running logs will save the loss_items in some internal state, it is recommended to
detach loss_items from their computational graph for memory efficiency.

### Using Your Own Loss - Training with Configuration Files

When using configuration files, for example, training using train_from_recipe (or similar, when the underlying train method that is being called is Trainer.train_from_config(...)),  In your ``my_loss.py``, register your loss class by decorating the class with `register_loss`:
```python
import torch.nn
from super_gradients.common.registry import register_loss
 
@register_loss("my_loss")
class MyLoss(torch.nn.Module):
    ...
```

Then, in your `my_training_hyperparams.yaml`, use `"my_loss"` in the same way as any other loss supported in SG:
```yaml
...

loss: my_loss

criterion_params:
  ...
```

Last, in your ``my_train_from_recipe_script.py`` file, just import the newly registered class (even though the class itself is unused, just to trigger the registry):
        
```python
from omegaconf import DictConfig
import hydra
import pkg_resources
from my_loss import MyLoss
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
