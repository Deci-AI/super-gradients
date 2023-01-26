# Losses in SG

SuperGradients provides multiple Loss function implementations for various tasks:

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

All of the above, are just string aliases for the underlying torch.nn.Module classes, implementing the specified loss functions.

##Basic Usage of Implemented Loss Functions in SG:

When using configuration files, for example training using train_from_recipe (or similar, when the underlying train method that is being called is Trainer.train_from_config(...)):
    

In your `my_yolox_training_hyperparams.yaml` file:
  ```yaml
    ...
    ...
    
    loss: yolox_loss
    
    criterion_params:
      strides: [8, 16, 32]  # output strides of all yolo outputs
      num_classes: 80
  ```
  
   `criterion_params` dictionary will be unpacked to the underlying `yolox_loss` class constructor.


Another usage case, is when using a direct Trainer.train(...) call:
   - In your `my_training_script.py`
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

## Using Your Own Loss
SuperGradients also supports user-defined loss functions assuming they are torch.nn.Module inheritors, and that their `forward` signature is in the form:

```python
import torch.nn

MyLoss(torch.nn.Module):
...
forward(preds, target):
...

```
And as the argument names suggest- the first argument is the model's output, and target is the label/ground truth (argument naming is arbitrary and does not need to be specifically 'preds' or 'target').
Loss functions accepting additional arguments in their `forward` method will be supported in the future.
When using configuration files, for example training using train_from_recipe (or similar, when the underlying train method that is being called is Trainer.train_from_config(...)),  In your ``my_loss.py``, register your loss class by decorating the class with `register_loss`:
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
Though not as convenient as using `register_loss`, one can also equivalently instantiate objects when using train_from_recipe (or similar, when the underlying train method is Trainer.train_from_config(...) as demonstrated below:


In your `my_training_hyperparams.yaml` file:
```yaml
  ...
  ...
  loss:
    _target_: torch.nn.CrossEntropy

```
  Note that when passing an instantiated loss object, `criterion_params` will be ignored.
