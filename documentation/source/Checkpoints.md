# Model Checkpoints in SG

The first question that arises is: what is a checkpoint ?

From the [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) documentation:


    When a model is training, the performance changes as it continues to see more data. It is a best practice to save the state of a model throughout the training process. This gives you a version of the model, a checkpoint, at each key point during the development of the model. Once training has completed, use the checkpoint that corresponds to the best performance you found during the training process.
    
    Checkpoints also enable your training to resume from where it was in case the training process is interrupted.

## Checkpoints Saving in SG: Which, When and Where?

From the previous subsection, we can understand that checkpoints saved at different times during training have different purposes.
That's why in SG multiple checkpoints are saved throughout training:

| Checkpoint Filename        | When is it saved?           |
| ------------- |:-------------:|
| `ckpt_best.pth`     | Each time we reach a new best [metric_to_watch](https://github.com/Deci-AI/super-gradients/blob/69d8d19813964022af192a34b6e7853edac34a75/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml#L39) when perfroming validation.|
| `ckpt_latest.pth`     | At the end of every epoch, constantly overriding. |
| `average_model.pth` | At the end of training - composed of 10 best models according to  [metric_to_watch](https://github.com/Deci-AI/super-gradients/blob/69d8d19813964022af192a34b6e7853edac34a75/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml#L39) and will only be save when the training_param `average_best_models`=True.     |
| `ckpt_epoch_{EPOCH_INDEX}.pth`| At the end of a fixed epoch number `EPOCH_INDEX` if it is specified through `save_ckpt_epoch_list` training_param |

#### Where are the checkpoint files saved ?

The checkpoint files will be saved at <PATH_TO_CKPT_ROOT_DIR>/experiment_name/.

The checkpoint root directory is controlled by the user and can be passed to `Trainer` constructor through the `ckpt_root_dir` argument.

When working with a cloned version of SG, one can leave out the `ckpt_root_dir` arg and checkpoints will be saved in the `checkpoints` module.



## Checkpoint Structure in SG

Checkpoints in SG are instances of [state_dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
Apart from the model weights, they hold additional information about the model's training.

The checkpoint keys:

"net": The network's state_dict (state_dict).

"acc": The network's achieved metric value on the validation set ([metric_to_watch in training_params](https://github.com/Deci-AI/super-gradients/blob/69d8d19813964022af192a34b6e7853edac34a75/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml#L39) (float).

"epoch": The last epoch performed.

"optimizer_state_dict": The state_dict of the optimizer (state_dict).

"scaler_state_dict": Optional - only present when training with [mixed_precision=True](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/average_mixed_precision.md). The state_dict of Trainer.scaler.

"ema_net": Optional - only present when training with [ema=True](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/EMA.md). The EMA model's state_dict. Note that `average_model.pth` lacks this entry even if ema=True since the average model's snapshots are of the EMA network already (i.e the "net" entry is already an average of the EMA snapshots).


## Loading Checkpoints

When discussing checkpoint loading in SG we must separate it into two use cases: loading weights and resuming training.
While the former requires the model's state_dict alone, SG checkpoint loading methods introduce more functionality than PyTorch's vanilla `load_state_dict()`, especially for SG trained checkpoints.

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
            "loss": "cross_entropy",
            "criterion_params": {},
            "save_ckpt_epoch_list": [10,15]
            ...
        }
        trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
   ```

Then at the end of training, our `ckpt_root_dir` contents will look similar to:

```
my_checkpoints_folder
|─── my_resnet18_training_experiment
│       ckpt_best.pth                     # Model checkpoint on best epoch
│       ckpt_latest.pth                   # Model checkpoint on last epoch
│       average_model.pth                 # Model checkpoint averaged over epochs
|       ckpt_epoch_10.pth                 # Model checkpoint of epoch 10
|       ckpt_epoch_15.pth                 # Model checkpoint of epoch 15
│       events.out.tfevents.1659878383... # Tensorflow artifacts of a specific run
│       log_Aug07_11_52_48.txt            # Trainer logs of a specific run
└─── some_other_training_experiment_name
        ...
```

Suppose we wish to load the weights from `ckpt_best.pth`, we can simply pass its path to the `checkpoint_path` argument in `models.get(...)`:

   ```python
        from super_gradients.training import models
        from super_gradients.common.object_names import Models
        
        model = models.get(model_name=Models.RESNET18, num_classes=10, checkpoint_path="/path/to/my_checkpoints_folder/my_resnet18_training_experiment/ckpt_best.pth")
   ```

`Important: when loading SG trained checkpoints using models.get(...), if the network was trained with EMA, the EMA weights will be the ones loaded. `

If we already possess an instance of our model, we can also directly use `load_checkpoint_to_model`:

   ```python
        from super_gradients.training import models
        from super_gradients.common.object_names import Models
        from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
        
        model = models.get(model_name=Models.RESNET18, num_classes=10)
        load_checkpoint_to_model(net=model, ckpt_local_path="/path/to/my_checkpoints_folder/my_resnet18_training_experiment/ckpt_best.pth")
   ```
### Extending the Functionality of PyTorch's `strict` Parameter in `load_state_dict()`

When not familiar with PyTorch's `strict` Parameter in `load_state_dict()`, please see [PyTorch's docs on this matter](https://pytorch.org/tutorials/beginner/saving_loading_models.html#id4) first.

The equivalent arguments  for PyTorch's `strict` Parameter in `load_state_dict()` in `models.get()` and `load_checkpoint_to_model` are `strict` and `strict_load` respectively, and expect SG's `StrictLoad` enum type.

Let's have a look at its possible values:

```python

class StrictLoad(Enum):
    """
    Wrapper for adding more functionality to torch's strict_load parameter in load_state_dict().
    Attributes:
        OFF              - Native torch "strict_load = off" behaviour. See nn.Module.load_state_dict() documentation for more details.
        ON               - Native torch "strict_load = on" behaviour. See nn.Module.load_state_dict() documentation for more details.
        NO_KEY_MATCHING  - Allows the usage of SuperGradient's adapt_checkpoint function, which loads a checkpoint by matching each
                           layer's shapes (and bypasses the strict matching of the names of each layer (ie: disregards the state_dict key matching)).
    """

    OFF = False
    ON = True
    NO_KEY_MATCHING = "no_key_matching"

```

In other words, we added another loading mode option- `no_key_matching`. This option exploits the fact that the `state_dicts` are `OrderedDict`s, and come in handy when the underlying network's structure remains the same but the `state_dict`s keys do not match ones inside the models `state_dict`.
Let's demonstrate the different strict modes with a simple example:

```python
import torch

class ModelA(torch.nn.Module):

```


### Loading Pretrained Weights from the SG Model Zoo

Using `models.get(...)`, you can load any of our pretrained models in 3 lines of code:
```python
        from super_gradients.training import models
        from super_gradients.common.object_names import Models
        
        model = models.get(Models.YOLOX_S, pretrained_weights="coco")
```

The `pretrained_weights` argument specifies the dataset on which the pretrained weights were trained on. [Here is the full list of pretrained weights](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/Computer_Vision_Models_Pretrained_Checkpoints.md).
