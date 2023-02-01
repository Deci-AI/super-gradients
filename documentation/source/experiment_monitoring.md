# Third-party experiment monitoring

SuperGradients supports out-of-the-box Weights & Biases (wandb) and ClearML. 
You can also inherit from our base class to integrate any monitoring tool with minimal code change.  

### Tensorboard
**requirements**: None

Tensorboard is natively integrated into the training and validation steps. You can find how to use it in [this section](TODO:add_link_to_our_tn_description).


### Weights & Biases
**requirements**:
- Install `wandb`
- Set up wandb according to the [official documentation](https://docs.wandb.ai/quickstart#1.-set-up-wandb)
- Make sure to login (You can check if you have a `~/.netrc` token)
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
        "entity": "<YOUR-ENTITY-NAME>",         # username or team name where you're sending runs
        "api_server": "<OPTIONAL-WANDB-URL>"    # Optional: In case your experiment tracking is not hosted at wandb servers
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


### Integrate any other Monitoring tool
If your favorite monitoring tool is not supported by SuperGradients, you can simply implement a class inheriting from `BaseSGLogger`
that you will then pass to the training parameters.

```python
import numpy as np
import torch
from PIL import Image
from typing import Union

from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.environment.ddp_utils import multi_process_safe


@register_logger()
class CustomSGLogger(BaseSGLogger):
    """Logger responsible to push logs and tensorboard artifacts to Deci platform."""

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        storage_location: str,
        resumed: bool,
        training_params: dict,
        checkpoints_dir_path: str,
    ):
        super().__init__(
            project_name=project_name,
            experiment_name=experiment_name,
            storage_location=storage_location,
            resumed=resumed,
            training_params=training_params,
            checkpoints_dir_path=checkpoints_dir_path,
        )
        self.my_client = ... # instantiate your monitoring tool client

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        super(CustomSGLogger, self).add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        
        self.my_client.write_metric(...) # Upload a scalar to your monitoring server
    
    @multi_process_safe
    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format="CHW", global_step: int = None):
        self.my_client.add_image(tag=tag, img_tensor=image, dataformats=data_format, global_step=global_step)

    ...
```
You can overwrite any method from `BaseSGLogger` to customize it to your need.
Then, you can pass it to your `training_params` exactly like WandB and ClearML.

```python
from super_gradients import Trainer

trainer = Trainer("experiment_name")
model = ...

training_params = {
    ...,                                                    # Your training params
    "sg_logger": "CustomSGLogger",                          # Your custom CustomSGLogger
    "sg_logger_params": {"project_name": "my_project_name"} # Params that will be passed to __init__ of your CustomSGLogger  
}

trainer.train(model=model, training_params=training_params, ...)
```

**Notes**:
 - `@multi_process_safe` prevents multiple training nodes to do the same action. Check out [DDP documentation](TODO) for more details
 - `@register_logger()` registers your class into our factory, allowing it to be instantiated from a string.
 - `sg_logger_params` only requires `project_name`, the rest is provided by the Trainer.


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


## Chose your monitoring tool in the recipes
You can update a [recipe](configuration_files.md) to use the monitoring tool you want by setting the sg_logger and sg_logger_params in (recipes/training_hyperparams)[TODO].

Here is an example for WandB;
```yaml
sg_logger: wandb_sg_logger, # Weights&Biases Logger, see class super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger for details
sg_logger_params:             # Params that will be passes to __init__ of the logger super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger
  project_name: project_name, # W&B project name
  save_checkpoints_remote: True,
  save_tensorboard_remote: True,
  save_logs_remote: True,
  entity: <YOUR-ENTITY-NAME>,         # username or team name where you're sending runs
  api_server: <OPTIONAL-WANDB-URL>    # Optional: In case your experiment tracking is not hosted at wandb servers
```
