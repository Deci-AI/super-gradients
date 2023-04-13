# Phase Callbacks

Integrating your own code into an already existing training pipeline can draw much effort on the user's end.
To tackle this challenge, a list of callables triggered at specific points of the training code can be passed through `phase_calbacks_list` inside `training_params` when calling `Trainer.train(...)`.

SG's `super_gradients.training.utils.callbacks` module implements some common use cases as callbacks:

    ModelConversionCheckCallback
    LRCallbackBase
    EpochStepWarmupLRCallback
    BatchStepLinearWarmupLRCallback
    StepLRCallback
    ExponentialLRCallback
    PolyLRCallback
    CosineLRCallback
    FunctionLRCallback
    LRSchedulerCallback
    DetectionVisualizationCallback
    BinarySegmentationVisualizationCallback
    TrainingStageSwitchCallbackBase
    YoloXTrainingStageSwitchCallback

For example, the YoloX's COCO detection training recipe uses `YoloXTrainingStageSwitchCallback` to turn off augmentations and incorporate L1 loss starting from epoch 285:

`super_gradients/recipes/training_hyperparams/coco2017_yolox_train_params.yaml`:

```yaml

max_epochs: 300
...

loss: yolox_loss

...

phase_callbacks:
  - YoloXTrainingStageSwitchCallback:
      next_stage_start_epoch: 285
...
```

Another example is how we use `BinarySegmentationVisualizationCallback` to visualize predictions during training in our [Segmentation Transfer Learning Notebook](https://bit.ly/3qKwMbe):


## Integrating Your Code with Callbacks

Integrating your code requires implementing a callback that `Trainer` would trigger in the proper phases inside SG's training pipeline.

So let's first get familiar with `super_gradients.training.utils.callbacks.base_callbacks.Callback` class.

It implements the following methods:

```python
    on_training_start(self, context: PhaseContext) -> None
    on_train_loader_start(self, context: PhaseContext) -> None:
    on_train_batch_start(self, context: PhaseContext) -> None:
    on_train_batch_loss_end(self, context: PhaseContext) -> None:
    on_train_batch_backward_end(self, context: PhaseContext) -> None:
    on_train_batch_gradient_step_start(self, context: PhaseContext) -> None:
    on_train_batch_gradient_step_end(self, context: PhaseContext) -> None:
    on_train_batch_end(self, context: PhaseContext) -> None:
    on_train_loader_end(self, context: PhaseContext) -> None:
    on_validation_loader_start(self, context: PhaseContext) -> None:
    on_validation_batch_start(self, context: PhaseContext) -> None:
    on_validation_batch_end(self, context: PhaseContext) -> None:
    on_validation_loader_end(self, context: PhaseContext) -> None:
    on_validation_end_best_epoch(self, context: PhaseContext) -> None:
    on_test_loader_start(self, context: PhaseContext) -> None:
    on_test_batch_start(self, context: PhaseContext) -> None:
    on_test_batch_end(self, context: PhaseContext) -> None:
    on_test_loader_end(self, context: PhaseContext) -> None:
    on_training_end(self, context: PhaseContext) -> None:

```

Our callback needs to inherit from the above class and override the appropriate methods according to the points at which we would like to trigger it.

To understand which methods we need to override, we need to understand better when are the above methods triggered.

From the class docs, the order of the events is as follows:
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

As you noticed, all of `Callback`'s methods expect a single argument - a `PhaseContext` instance.
This argument gives access to some variables at the points mentioned above in the code through its attributes.
We can discover what variables are exposed by looking at the documentation of the `Callback`'s specific methods we need to override.

For example:
```python
...
    def on_training_start(self, context: PhaseContext) -> None:
        """
        Called once before the start of the first epoch
        At this point, the context argument is guaranteed to have the following attributes:
        - optimizer
        - net
        - checkpoints_dir_path
        - criterion
        - sg_logger
        - train_loader
        - valid_loader
        - training_params
        - checkpoint_params
        - architecture
        - arch_params
        - metric_to_watch
        - device
        - ema_model
        ...
        :return:
        """
```

Now let's implement our callback.
Suppose we would like to implement a simple callback that saves the first batch of images in each epoch for both training and validation
in a new folder called "batch_images" under our local checkpoints directory.


Our callback needs to be triggered in 3 places:
1. At the start of training, create a new "batch_images" under our local checkpoints directory.
2. Before passing a train image batch through the network.
3. Before passing a validation image batch through the network.

Therefore, our callback will override `Callback`'s `on_training_start`, `on_train_batch_start`, and `on_validation_batch_start` methods:

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

Note the `@multi_process_safe` decorator, which allows the callback to be triggered precisely once when running distributed training.

For coded training scripts (i.e., not [using configuration files](configuration_files.md)), we can pass an instance of the callback through `phase_callbacks`:

   ```python
...

...
trainer = Trainer("my_experiment")
train_dataloader = ...
valid_dataloader = ...
model = ...

train_params = {
    ...
    "loss": "cross_entropy",
    "criterion_params": {}
    ...
    "phase_callbacks": [SaveFirstBatchCallback()],
}

trainer.train(training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```

Otherwise, for training with configuration files, we need to register our new callback by decorating it with the `register_loss` decorator:

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

Last, in your ``my_train_from_recipe_script.py`` file, import the newly registered class (even though the class itself is unused, just to trigger the registry):
        
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
