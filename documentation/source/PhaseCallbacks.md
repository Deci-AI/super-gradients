# Phase Callbacks

Integrating your own code into an already existing training pipeline can draw much effort on the user's end.
To tackle this challenge, a list of callables triggered at specific points of the training code can
be passed through `training_params.phase_calbacks_list` when calling `Trainer.train(...)`.

SG's `super_gradients.training.utils.callbacks` module implements some common use cases as callbacks:

    ModelConversionCheckCallback
    LRCallbackBase
    LinearEpochLRWarmup
    LinearBatchLRWarmup
    StepLRScheduler
    ExponentialLRScheduler
    PolyLRScheduler
    CosineLRScheduler
    FunctionLRScheduler
    LRSchedulerCallback
    DetectionVisualizationCallback
    BinarySegmentationVisualizationCallback
    TrainingStageSwitchCallbackBase
    YoloXTrainingStageSwitchCallback

For example, the YoloX's COCO detection training recipe uses `YoloXTrainingStageSwitchCallback` to turn 
off augmentations and incorporate L1 loss starting from epoch 285:

`super_gradients/recipes/training_hyperparams/coco2017_yolox_train_params.yaml`:

```yaml
max_epochs: 300
...

loss: YoloXDetectionLoss

...

phase_callbacks:
  - YoloXTrainingStageSwitchCallback:
      next_stage_start_epoch: 285
...
```

Another example is how we use `BinarySegmentationVisualizationCallback` to visualize predictions
during training in the [Segmentation Transfer Learning Notebook](https://bit.ly/3qKwMbe):


### How Callbacks work

`Callback` implements the following methods:

```python
# super_gradients.training.utils.callbacks.base_callbacks.Callback
class Callback:
    def on_training_start(self, context: PhaseContext) -> None: pass
    def on_train_loader_start(self, context: PhaseContext) -> None: pass
    def on_train_batch_start(self, context: PhaseContext) -> None: pass
    def on_train_batch_loss_end(self, context: PhaseContext) -> None: pass
    def on_train_batch_backward_end(self, context: PhaseContext) -> None: pass
    def on_train_batch_gradient_step_start(self, context: PhaseContext) -> None: pass
    def on_train_batch_gradient_step_end(self, context: PhaseContext) -> None: pass
    def on_train_batch_end(self, context: PhaseContext) -> None: pass
    def on_train_loader_end(self, context: PhaseContext) -> None: pass
    def on_validation_loader_start(self, context: PhaseContext) -> None: pass
    def on_validation_batch_start(self, context: PhaseContext) -> None: pass
    def on_validation_batch_end(self, context: PhaseContext) -> None: pass
    def on_validation_loader_end(self, context: PhaseContext) -> None: pass
    def on_validation_end_best_epoch(self, context: PhaseContext) -> None: pass
    def on_test_loader_start(self, context: PhaseContext) -> None: pass
    def on_test_batch_start(self, context: PhaseContext) -> None: pass
    def on_test_batch_end(self, context: PhaseContext) -> None: pass
    def on_test_loader_end(self, context: PhaseContext) -> None: pass
    def on_training_end(self, context: PhaseContext) -> None: pass
```

The order of the events is as follows:
```python
on_training_start(context)                              # called once before training starts, good for setting up the warmup LR

    for epoch in range(epochs):
        on_train_loader_start(context)
            for batch in train_loader:
                on_train_batch_start(context)
                on_train_batch_loss_end(context)               # called after loss has been computed
                on_train_batch_backward_end(context)           # called after .backward() was called
                on_train_batch_gradient_step_start(context)    # called before the optimizer step about to happen (gradient clipping, logging of gradients)
                on_train_batch_gradient_step_end(context)      # called after gradient step was done, good place to update LR (for step-based schedulers)
                on_train_batch_end(context)
        on_train_loader_end(context)

        on_validation_loader_start(context)
            for batch in validation_loader:
                on_validation_batch_start(context)
                on_validation_batch_end(context)
        on_validation_loader_end(context)
        on_validation_end_best_epoch(context)

    on_test_start(context)
        for batch in test_loader:
            on_test_batch_start(context)
            on_test_batch_end(context)
    on_test_end(context)

on_training_end(context)                    # called once after training ends.
```

Callbacks are implemented by inheriting this `Callback` class, and then by override any of the above-mentioned 
method with the wanted behavior. 

### Phase Context

You may have noticed that the `Callback`'s methods expect a single argument - a `PhaseContext` instance.

`PhaseContext` includes attributes representing a wide range of training attributes at a given point of the training.

```
    - epoch
    - batch_idx
    - optimizer
    - metrics_dict
    - inputs
    - preds
    - target
    - metrics_compute_fn
    - loss_avg_meter
    - loss_log_items
    - criterion
    - device
    - experiment_name
    - ckpt_dir
    - net
    - lr_warmup_epochs
    - sg_logger
    - train_loader
    - valid_loader
    - test_loader
    - training_params
    - ddp_silent_mode
    - checkpoint_params
    - architecture
    - arch_params
    - metric_to_watch
    - valid_metrics
    - ema_model
    - loss_logging_items_names
```

Each of these attributes is set to `None` by default, up until the point it computed or defined in the training pipeline.
- E.g. `epoch` will be `None` within `on_training_start` because, as explained above, this steps happens before the first epoch begins

You can find which context attribute is set by looking into each method docstring:
```python
class Callback:
    
    ...
    
    def on_training_start(self, context: PhaseContext) -> None:
        """
        Called once before start of the first epoch
        At this point, the context argument will have the following attributes:
            - optimizer
            - criterion
            - device
            - experiment_name
            - ckpt_dir
            - net
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics

        The corresponding Phase enum value for this event is Phase.PRE_TRAINING.
        :param context:
        """
        pass
```

### Build your own Callback

Suppose we would like to implement a simple callback that saves the first batch of images in each epoch for both 
training and validation in a new folder called "batch_images" under the local checkpoints directory.

This callback needs to be triggered in 3 places:
1. At the start of training, create a new "batch_images" under the local checkpoints directory.
2. Before passing a train image batch through the network, save it in the new folder.
3. Before passing a validation image batch through the network, save it in the new folder.

Therefore, the callback will override `Callback`'s `on_training_start`, `on_train_batch_start`, and `on_validation_batch_start` methods:

```python
from super_gradients.training.utils.callbacks import Callback, PhaseContext
from super_gradients.common.environment.ddp_utils import multi_process_safe
import os
from torchvision.utils import save_image


class SaveFirstBatchCallback(Callback):
    def __init__(self):
        self.outputs_path = None
        self.saved_first_validation_batch = False

    @multi_process_safe
    def on_training_start(self, context: PhaseContext) -> None:
        outputs_path = os.path.join(context.ckpt_dir, "batch_images")
        os.makedirs(outputs_path, exist_ok=True)

    @multi_process_safe
    def on_train_batch_start(self, context: PhaseContext) -> None:
        if context.batch_idx == 0:
            save_image(context.inputs, os.path.join(self.outputs_path, f"first_train_batch_epoch_{context.epoch}.png"))

    @multi_process_safe
    def on_validation_batch_start(self, context: PhaseContext) -> None:
        if context.batch_idx == 0 and not self.saved_first_validation_batch:
            save_image(context.inputs, os.path.join(self.outputs_path, f"first_validation_batch_epoch_{context.epoch}.png"))
            self.saved_first_validation_batch = True
```
**IMPORTANT**

When training on multiple nodes (see [DDP](device.md)), the callback will be called at each step once for every 
node you are working with. This behaviour may be useful in some specific cases, but in general you will 
want to have each method to be triggered only once per step. You can add the decorator `@multi_process_safe` to ensure 
that only the main node will trigger the callback. 

In our example, we want to trigger only once per step, so we need to add the `@multi_process_safe` decorator.

### Using Custom Callback within Python Script
The callback can directly be passed through `training_params.phase_callbacks`

```python
trainer = Trainer("my_experiment")
train_dataloader = ...
valid_dataloader = ...
model = ...

train_params = {
    "loss": "CrossEntropyLoss",
    "criterion_params": {},
    "phase_callbacks": [SaveFirstBatchCallback()],
    ...
}

trainer.train(training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```

### Using Custom Callback in a Recipe
If you are working with [Configuration files](configuration_files.md), you will be required to do an extra step.
This is similar to using any custom objects in a recipe, and is already defined in the [above-mentioned](configuration_files.md). 

To summarize, you need to register the new callback by decorating it with the `register_callback` decorator, 
so that SuperGradients would know how to instantiate it from the `.yaml` recipe.

```python
from super_gradients.training.utils.callbacks import Callback, PhaseContext
from super_gradients.common.environment.ddp_utils import multi_process_safe
import os
from torchvision.utils import save_image
from super_gradients.common.registry.registry import register_callback

@register_callback()
class SaveFirstBatchCallback(Callback):
    def __init__(self):
        self.outputs_path = None
        self.saved_first_validation_batch = False

    @multi_process_safe
    def on_training_start(self, context: PhaseContext) -> None:
        outputs_path = os.path.join(context.ckpt_dir, "batch_images")
        os.makedirs(outputs_path, exist_ok=True)

    @multi_process_safe
    def on_train_batch_start(self, context: PhaseContext) -> None:
        if context.batch_idx == 0:
            save_image(context.inputs, os.path.join(self.outputs_path, f"first_train_batch_epoch_{context.epoch}.png"))

    @multi_process_safe
    def on_validation_batch_start(self, context: PhaseContext) -> None:
        if context.batch_idx == 0 and not self.saved_first_validation_batch:
            save_image(context.inputs, os.path.join(self.outputs_path, f"first_validation_batch_epoch_{context.epoch}.png"))
            self.saved_first_validation_batch = True


```


Then, in your `my_training_hyperparams.yaml`, use `SaveFirstBatchCallback` in the same way as any other phase callback supported in SG:
  ```yaml
defaults:
  - default_train_params

max_epochs: 250

...
phase_callbacks:
  - SaveFirstBatchCallback
```

Last, make sure to import `SaveFirstBatchCallback` in the script you use to launch training from config:
        
```python

  from omegaconf import DictConfig
  import hydra
  import pkg_resources
  from my_callbacks import SaveFirstBatchCallback
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

This is required, as otherwise `SaveFirstBatchCallback` would not be imported at all and therefore SuperGradients 
would fail to recognize and instantiate it. 
