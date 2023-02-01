# Third-party experiment monitoring

SuperGradients supports out-of-the-box Weights & Biases (wandb) and ClearML. 


Simply update your train parameters to specify your preferred third-party logging tool, and the Trainer will 
take care of the rest, handling all the necessary logic and integration.

### Tensorboard
**requirements**: None

Tensorboard is natively integrated into the training and validation steps. You can find how to use it in [this section](TODO:add_link_to_our_tn_description).


### Weights & Biases
**requirements**:
- Install `wandb`
- Set up wandb according to the [official documentation](https://docs.wandb.ai/quickstart#1.-set-up-wandb)
- Adapt your code like in the following example

```python
from super_gradients import Trainer

trainer = Trainer("experiment_name")
model = ...

training_params = {
    ...                             # Your training params
    "sg_logger": "wandb_sg_logger", # Weights&Biases Logger, see class super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger for details
    "sg_logger_params":             # Params that will be passes to __init__ of the logger super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger
      {
        "project_name": "project_name", # W&B project name
        "save_checkpoints_remote": True,
        "save_tensorboard_remote": True,
        "save_logs_remote": True,
      }
}

trainer.train(model=model, training_params=training_params, ...)
```


### ClearML
**requirements**:
- Install `clearml` 
- Set up CleaML according to the [official documentation](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)
- Adapt your code like in the following example

```python
from super_gradients import Trainer

trainer = Trainer("experiment_name")
model = ...

training_params = {
    ...                                 # Your training params
    "sg_logger": "clearml_sg_logger",   # ClearML Logger, see class super_gradients.common.sg_loggers.wandb_sg_logger.ClearMLSGLogger for details
    "sg_logger_params":                 # Params that will be passes to __init__ of the logger super_gradients.common.sg_loggers.wandb_sg_logger.ClearMLSGLogger 
      {
        "project_name": "project_name", # ClearML project name
        "save_checkpoints_remote": True,
        "save_tensorboard_remote": True,
        "save_logs_remote": True,
      } 
}

trainer.train(model=model, training_params=training_params, ...)
```


## Uploading custom objects with a callback
Callbacks are the way to go when it comes to inserting small pieces of code into the training/validation loop of SuperGradients.
For more information, please check out our tutorial on [how to use callbacks in SuperGradients](TODO:add_link)

Here is a short example of how sg_logger can be used in callbacks:
```python
from super_gradients.training.utils.callbacks.base_callbacks import PhaseContext, Callback

def do_something(inputs, target, preds):
    pass

class DetectionVisualizationCallback2(Callback):
    """Save a custom metric to tensorboard and wandb/clearml"""

    def __init__(self):
        super(Callback, self).__init__()

    def on_validation_batch_end(self, context: PhaseContext) -> None:

        # Do something using the PhaseContext
        custom_metric = do_something(context.inputs, context.target, context.preds)
        
        # Save it to the tensorboard and wandb/clearml
        context.sg_logger.add_scalar(
            tag="custom_metric",
            scalar_value=custom_metric,
            global_step=context.epoch,
        )
```

The sg_logger can also be used to upload files, text, images, checkpoints, ...

We encourage you to check out the API documentation of `super_gradients.common.sg_loggers.base_sg_logger.BaseSGLogger` to see every available method.
