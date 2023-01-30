# Local Logging

SuperGradients automatically logs locally multiple files that can help you explore your experiments results. This includes one tensorboard and 3 .txt files.



## I. Tensorboard logging
To easily keep track of your experiments, SuperGradients saves your results in `events.out.tfevents` format that can be used by tensorboard.

**What does it include?** This tensorboard includes all of your training and validation metrics but also other information such as learning rate, system metrics (CPU, GPU, ...), and more.

**Where is it saved?** `<ckpt_root_dir>/<experiment_name>/events.out.tfevents.<unique_id>`

**How to launch?**`tensorboard --logdir checkpoint_path/events.out.tfevents.<unique_id>`



## II. Experiment logging
In case you cannot launch a tensorboard instance, you can still find a summary of your experiment saved in a readable .txt format.

**What does it include?** The experiment configuration and training/validation metrics.

**Where is it saved?** `<ckpt_root_dir>/<experiment_name>/experiment_logs_<date>.txt`




## III. Console logging
For better debugging and understanding of past runs, SuperGradients gathers all the print statements and logs into a 
local file, providing you the convenience to review console outputs of any experiment at any time.

**What does it include?** All the prints and logs that were displayed on the console, but not the filtered logs.

**Where is it saved?**
- Upon importing SuperGradients, console outputs and logs will be stored in `~/sg_logs/console.log`.
- When instantiating the super_gradients.Trainer, all console outputs and logs will be redirected to the experiment folder `<ckpt_root_dir>/<experiment_name>/console_<date>.txt`.

**How to set log level?** You can filter the logs displayed on the console by setting `CONSOLE_LOG_LEVEL=<LOG-LEVEL> # DEBUG/INFO/WARNING/ERROR`



## IV. Loggers logging
Contrary to the console logging, the logger logging is restricted to the loggers messages (such as `logger.log`, `logger.info`, ...).
This means that it includes any log that was under the logging level (`logging.DEBUG` for instance), but not the prints.

**What does it include?** Anything logged with a logger (`logger.log`, `logger.info`, ...), even the filtered logs.

**Where is it saved?** `<ckpt_root_dir>/<experiment_name>/logs_<date>.txt`

**How to set log level?** You can filter the logs saved in the file by setting `FILE_LOG_LEVEL=<LOG-LEVEL> # DEBUG/INFO/WARNING/ERROR`


## Other


#### Environment Sanity Check
SuperGradients automatically checks compatibility between the installed libraries and the required ones.
It will log an error - but not stop the code - for each library that was installed with a version lower than required.
For libraries with version higher than required, this information will just be logged at a DEBUG level.



#### Crash Tip
It can sometimes be very time consuming to debug an exceptions when the error raised is not explicit.
To avoid this, SuperGradients implemented a Crash Tip system that decorates errors raised from different libraries to help you fix the issue.

**Example**
The error raised by hydra when you made an indentation error is hard to understand (see topmost RuntimeError).
Under the exception, SuperGradients prints a Crash Tip that explains what went wrong, and how to fix it.
![crash_tip.png](crash_tip.png)

The number of crash tips is limited to cases that were faced by the community, so if you face an exception that is hard to understand feel free to share with us!

**How to disable?** The Crash tip can be shut down by setting the environment variable `CRASH_HANDLER=FALSE`.


---
TOREMOVE

## Third party logging
SuperGradients supports out-of-the-box Weights & Biases (wandb) and ClearML. 

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
