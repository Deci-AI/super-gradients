# Model Checkpoints

*If you are not familiar on how experiments are managed, you can check [this tutorial](experiment_management.md)*


The first question that arises is: what is a checkpoint?

From the [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) documentation:


    When a model is training, the performance changes as it sees more data. It is a best practice to save the state of a model throughout the training process. This gives you a version of the model, a checkpoint, at each key point during the development of the model. Once training has been completed, use the checkpoint corresponding to the best performance you found during the training process.
    
    Checkpoints also enable your training to resume where it was in case the training process is interrupted.

## Checkpoints Saving: Which, When, and Where?

From the previous subsection, we understand that checkpoints saved at different times during training have different purposes.
That's why in SG, multiple checkpoints are saved throughout training:

| Checkpoint Filename | When is it saved?|
| ------------- |:-------------:|
| `ckpt_best.pth` | Each time we reach a new best [metric_to_watch](https://github.com/Deci-AI/super-gradients/blob/69d8d19813964022af192a34b6e7853edac34a75/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml#L39) when performing validation. |
| `ckpt_latest.pth` | At the end of every epoch, constantly overriding. |
| `average_model.pth` | At the end of training - composed of 10 best models according to  [metric_to_watch](https://github.com/Deci-AI/super-gradients/blob/69d8d19813964022af192a34b6e7853edac34a75/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml#L39) and will only be save when the training_param `average_best_models`=True. |
| `ckpt_epoch_{EPOCH_INDEX}.pth` | At the end of a fixed epoch number `EPOCH_INDEX` if it is specified through `save_ckpt_epoch_list` training_param |

#### Where are the checkpoint files saved?

The checkpoint files will be saved at `<ckpt_root_dir>/<experiment_name>/<run_dir>`.

- `ckpt_root_dir` and `experiment_name` can be set by the user when instantiating the `Trainer`.
```python
Trainer(ckpt_root_dir='path/to/ckpt_root_dir', experiment_name="my_experiment")
```
- `run_dir` is unique and automatically generated each time you start a new training, with `trainer.train(...)`

When working with a cloned version of SG, one can leave out the `ckpt_root_dir` arg, and checkpoints will be saved under the `super_gradients/checkpoints` directory.

## Checkpoint Structure

Checkpoints in SG are instances of [state_dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
They hold additional information about the model's training besides the model weights.

The checkpoint keys:

- `net`: The network's state_dict (state_dict).
- `acc`: The network's achieved metric value on the validation set ([metric_to_watch in training_params](https://github.com/Deci-AI/super-gradients/blob/69d8d19813964022af192a34b6e7853edac34a75/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml#L39) (float).
- `epoch`: The last epoch performed.
- `optimizer_state_dict`: The state_dict of the optimizer (state_dict).
- `scaler_state_dict`: Optional - only present when training with [mixed_precision=True](average_mixed_precision.md). The state_dict of Trainer.scaler.
- `ema_net`: Optional - only present when training with [ema=True](EMA.md). The EMA model's state_dict. Note that `average_model.pth` lacks this entry even if ema=True since the average model's snapshots are of the EMA network already (i.e., the "net" entry is already an average of the EMA snapshots).
- `torch_scheduler_state_dict`: Optional, will only be present when using a torch native lr scheduler (see [LRScheduling](LRScheduling.md))
## Remote Checkpoint Saving with SG Loggers

SG supports remote checkpoint saving using 3rd party tools (for example, [Weights & Biases](https://www.google.com/aclk?sa=l&ai=DChcSEwi1iaLxhYj9AhXejWgJHZYqCGIYABAAGgJ3Zg&sig=AOD64_30zInAUka20YKKdULr8PHnLnLWgg&q&adurl&ved=2ahUKEwiKxZvxhYj9AhUzTKQEHSJwCkcQ0Qx6BAgGEAE)).
To do so, specify `save_checkpoints_remote=True` inside `sg_logger_params` training_param.
See our documentation on [Third-party experiment monitoring](experiment_monitoring.md).


## Loading Checkpoints

When discussing checkpoint loading in SG, we must separate it into two use cases: loading weights and resuming training.
While the former requires the model's state_dict alone, SG checkpoint loading methods introduce more functionality than PyTorch's vanilla `load_state_dict()`, especially for SG-trained checkpoints.

### Loading Model Weights from a Checkpoint

Loading model weights can be done right after model initialization, using `models.get(...)`, or by explicitly calling `load_checkpoint_to_model` on the `torch.nn.Module` instance.

Suppose we have launched a training experiment with a similar structure to the one below:

```python
from super_gradients.training import Trainer
...
...
from super_gradients.training import models
from super_gradients.common.object_names import Models

trainer = Trainer("my_resnet18_training_experiment", ckpt_root_dir="/path/to/my_checkpoints_folder")
train_dataloader = ...
valid_dataloader = ...
model = models.get(model_name=Models.RESNET18, num_classes=10)

train_params = {
    ...
    "loss": "CrossEntropyLoss",
    "criterion_params": {},
    "save_ckpt_epoch_list": [10,15]
    ...
}
trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```

Then at the end of the training, our `ckpt_root_dir` contents will look similar to the following:

```
<ckpt_root_dir>
│
├── <experiment_name>
│   │
│   ├─── <run_dir>
│   │     ├─ ckpt_best.pth                   # Best performance during validation
│   │     ├─ ckpt_latest.pth                 # End of the most recent epoch
│   │     ├─ average_model.pth               # Averaged over specified epochs
│   │     ├─ ckpt_epoch_*.pth                # Checkpoints from specific epochs (like epoch 10, 15, etc.)
│   │     ├─ events.out.tfevents.*           # Tensorflow run artifacts
│   │     └─ log_<timestamp>.txt             # Trainer logs of the specific run
│   │
│   └─── <other_run_dir>
│        └─ ...
│
└─── <other_experiment_name>
    │
    ├─── <run_dir>
    │     └─ ...
    │
    └─── <another_run_dir>
          └─ ...
```

Suppose we wish to load the weights from `ckpt_best.pth`. We can simply pass its path to the `checkpoint_path` argument in `models.get(...)`:

```python
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(model_name=Models.RESNET18, num_classes=10, checkpoint_path="/path/to/my_checkpoints_folder/my_resnet18_training_experiment/RUN_20230802_131052_651906/ckpt_best.pth")
```

> Important: when loading SG-trained checkpoints using models.get(...), if the network was trained with EMA, the EMA weights will be the ones loaded. 

If we already possess an instance of our model, we can also directly use `load_checkpoint_to_model`:

```python
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model

model = models.get(model_name=Models.RESNET18, num_classes=10)
load_checkpoint_to_model(net=model, ckpt_local_path="/path/to/my_checkpoints_folder/my_resnet18_training_experiment/RUN_20230802_131052_651906/ckpt_best.pth")
```

### Extending the Functionality of PyTorch's `strict` Parameter in `load_state_dict()`

When not familiar with PyTorch's `strict` parameter in `load_state_dict()`, please see [PyTorch's docs on this matter](https://pytorch.org/tutorials/beginner/saving_loading_models.html#id4) first.

The equivalent arguments for PyTorch's `strict` parameter in `load_state_dict()` in `models.get()` and `load_checkpoint_to_model` are `strict` and `strict_load` respectively, and expect SG's `StrictLoad` enum type.

Let's have a look at its possible values:

```python

class StrictLoad(Enum):
    """
    Wrapper for adding more functionality to torch's strict_load parameter in load_state_dict().
    Attributes:
        OFF              - Native torch "strict_load = off" behavior. See nn.Module.load_state_dict() documentation for more details.
        ON               - Native torch "strict_load = on" behavior. See nn.Module.load_state_dict() documentation for more details.
        NO_KEY_MATCHING  - Allows the usage of SuperGradient's adapt_checkpoint function, which loads a checkpoint by matching each
                           layer's shapes (and bypasses the strict matching of the names of each layer (i.e., disregards the state_dict key matching)).
    """

    OFF = False
    ON = True
    NO_KEY_MATCHING = "no_key_matching"

```

In other words, we added another loading mode option- `no_key_matching`. This option exploits the fact that the `state_dicts` are `OrderedDict`s, and comes in handy when the underlying network's structure remains the same, but the `state_dict`s keys do not match the ones inside the models `state_dict`.
Let's demonstrate the different strict modes with a simple example:

```python
import torch

class ModelA(torch.nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

class ModelB(torch.nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.CONV2 = torch.nn.Sequential([torch.nn.Conv2d(6, 16, 5)])
        
```

Notice the above networks have identical weight structures but will have different keys in their `state_dict`s.
This is why loading a checkpoint from either one to the other, using `strict=True`, will fail and crash. Using `strict=False` will not crash and successfully load the first layer's weights only.
Using SG's `no_key_matching` will successfully load a checkpoint from either one to the other.

### Loading Pretrained Weights from the Model Zoo

Using `models.get(...)`, you can load any of our pre-trained models in 3 lines of code:
```python
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLOX_S, pretrained_weights="coco")
```

The `pretrained_weights` argument specifies the dataset on which the pre-trained weights were trained. [Here is the complete list of pre-trained weights](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/Computer_Vision_Models_Pretrained_Checkpoints.md).


### Loading Checkpoints: Training with Configuration Files

Prerequisites: [Training with Configuration Files](configuration_files.md)

Recall the SGs recipes library structure:


The `super_gradients/recipes` include the following subdirectories

- arch_params - containing configuration files for instantiating different models
- checkpoint_params - containing configuration files that define the loaded and saved checkpoints parameters for the training
- conversion_params - containing configuration files for the model conversion scripts (for deployment)
- dataset_params - containing configuration files for instantiating different datasets and dataloaders
- training_hyperparams - containing configuration files holding hyper-parameters for specific recipes

And now, let's take a look at the default parameters in `checkpoint_params`:

```yaml
load_backbone: False # whether to load only the backbone part of the checkpoint
checkpoint_path: # checkpoint path that is located in super_gradients/checkpoints
strict_load: True # key matching strictness for loading checkpoint's weights
pretrained_weights: # a string describing the dataset of the pre-trained weights (for example, "imagenent").
```

And note the above parameters are used to start the training with different weights (fine-tuning etc.) - they are passed to model.get() in the underlying flow of `Trainer.train_from_config(...)`:
```python

@classmethod
def train_from_config(cls, cfg: Union[DictConfig, dict]) -> Tuple[nn.Module, Tuple]:
    ...

    # BUILD NETWORK
    model = models.get(
        ...
        strict_load=cfg.checkpoint_params.strict_load,
        pretrained_weights=cfg.checkpoint_params.pretrained_weights,
        checkpoint_path=cfg.checkpoint_params.checkpoint_path,
        load_backbone=cfg.checkpoint_params.load_backbone,
    )

    # INSTANTIATE DATA LOADERS

    train_dataloader = ...
    val_dataloader = ...
    
    ...

    # TRAIN
    res = trainer.train(...)
... 
```

## Resuming Training

Resuming training in SG is a comprehensive process, controlled by three primary parameters that allow flexibility 
in continuing or branching off from specific training checkpoints. 
These parameters are used within `training_params`: `resume`, `run_id`, and `resume_path`.

```yaml
resume: False  # Option to continue training from the latest checkpoint.
run_id:        # ID to resume from a specific run within the same experiment.
resume_path:   # Direct path to a specific checkpoint file (.pth) to resume training.

...
```


#### 1. Resuming the Latest Run

By setting `resume=True`, SuperGradients will resume training from the last checkpoint within the same experiment.

Example:

```shell
# Continues from the latest run in the cifar_experiment.
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet experiment_name=cifar_experiment training_hyperparams.resume=True
```

#### 2. Resuming a Specific Run

Using `run_id`, you can resume training from a specific run within the same experiment, identified by the run ID.

Example:

```shell
# Continues from a specific run identified by the ID within cifar_experiment.
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet experiment_name=cifar_experiment run_id=RUN_20230802_131052_651906
```

#### 3. Branching off from a specific checkpoint

By specifying a `resume_path`, SuperGradients will create a new run directory, 
allowing training to resume from that specific checkpoint, 
and subsequently save the new checkpoints in this new directory.

Example:

```shell
# Branches from a specific checkpoint, creating a new run.
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet experiment_name=cifar_experiment training_hyperparams.resume_path=/path/to/checkpoint.pth
```

#### 4. Resuming with original recipe
Resuming is parameter dependant - you cannot resume the training of a model if there is a mismatch between the model architecture defined in your recipe, and the one in your checkpoint.

Therefore, if you trained a model a while ago, and that in the meantime you changed the model architecture definition,
then you won't be able to resume its training, loading the model would simply raise an exception.


To avoid this issue, SuperGradients provides an option to resume a training based on the recipe that was originally used to train the model. 

```
Trainer.resume_experiment(ckpt_root_dir=..., experiment_name=..., run_id=...)
```

- `run_id` is optional. You can use it to chose which run you want to resume. By default, it will resume the latest run of your experiment.

Note that  `Trainer.resume_experiment` can only resume training that were launched with `Trainer.train_from_config`.

See usage in our [resume_experiment_example](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/resume_experiment_example/resume_experiment.py).

## Resuming Training from SG Logger's Remote Storage (WandB only)

SG supports saving checkpoints throughout the training process in the remote storage defined by `SG Logger` (more info about this object and it's role during training in SG at [Third-party experiment monitoring](experiment_monitoring.md).)
Suppose we run an experiment with a `WandB` SG logger, then our `training_hyperparams` should hold:
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

The `save_checkpoints_remote` flag is set which will result in saving checkpoints in WandB throughout training.
Now, in case the training was interrupted, we can resume it from the checkpoint located in the WandB run storage by setting 2 training hyperparameters:
1. Set `resume_from_remote_sg_logger`:
```yaml
resume_from_remote_sg_logger: True
```
2. Pass `run_id` through  `wandb_id` to `sg_logger_params`:
```yaml
sg_logger: wandb_sg_logger, # Weights&Biases Logger, see class super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger for details
sg_logger_params:             # Params that will be passes to __init__ of the logger super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger
  wandb_id: <YOUR_RUN_ID>
  project_name: project_name, # W&B project name
  save_checkpoints_remote: True,
  save_tensorboard_remote: True,
  save_logs_remote: True,
  entity: <YOUR-ENTITY-NAME>,         # username or team name where you're sending runs
  api_server: <OPTIONAL-WANDB-URL>    # Optional: In case your experiment tracking is not hosted at wandb servers
```

And that's it! Once you re-launch your training, `ckpt_latest.pth` (by default) will be downloaded to the checkpoints directory, and the training will resume from it just as if it was locally stored.

## Evaluating Checkpoints

Analogically to the previous section, we often want to evaluate a checkpoint seamlessly without being familiar with the training configuration.
For this reason, SG introduces two methods: `Trainer.evaluate_checkpoint(...)` and `Trainer.evaluate_recipe(...)` and play similar roles to the two previous ways of resuming experiments suggested in the last section:

`Trainer.evaluate_checkpoint` is used to evaluate a checkpoint resulting from one of your previous experiments, using the same parameters (dataset, valid_metrics,...) as used during the training of the experiment.
`Trainer.evaluate_recipe`  is used to evaluate a checkpoint from SGs pre-trained model zoo or to evaluate a checkpoint with different parameters.
See both usages and documentation in the corresponding scripts [evaluate_checkpoint](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/evaluate_checkpoint_example/evaluate_checkpoint.py) and [evaluate_recipe](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/evaluate_checkpoint_example/evaluate_checkpoint.py).
