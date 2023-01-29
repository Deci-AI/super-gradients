# SuperGradient Logging system

## Console logging
For better debugging and understanding of past runs, SuperGradients gathers all the print statements and logs into a 
local file, providing you the convenience to review console outputs of any experiment at any time.

**Where is it saved?**
- Upon importing SuperGradients, console outputs and logs will be stored in `~/sg_logs/console.log`.
- When instantiating the super_gradients.Trainer, all console outputs and logs will be redirected to the experiment folder `/<path>/<to>/<experiments>/<experiment_name>/sg_logs_<date>.log`.


## Third party logging
With SuperGradients, you can easily track and log your results using Weights & Biases (wandb) or ClearML. 

Simply update your train parameters to specify your preferred third-party logging tool, and the Trainer will 
take care of the rest, handling all the necessary logic and integration.

### Weights & Biases
**requirements**:
- Install wandb
- Set up wandb according to the [official documentation](https://docs.wandb.ai/quickstart#1.-set-up-wandb)
- Adapt your code like in the following example

```python
from super_gradients import Trainer

# create a trainer object, look the declaration for more parameters
trainer = Trainer("experiment_name")
model = ...

training_params = {
    ...                             # Your training params
    "sg_logger": "wandb_sg_logger", # Weights&Biases Logger, see class WandBSGLogger for details
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
- Install clearml
- Set up cleaml according to the [official documentation](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)
- Adapt your code like in the following example

```python
from super_gradients import Trainer

# create a trainer object, look the declaration for more parameters
trainer = Trainer("experiment_name")
model = ...

training_params = {
    ...                                 # Your training params
    "sg_logger": "clearml_sg_logger",   # ClearML Logger, see class WandBSGLogger for details
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
