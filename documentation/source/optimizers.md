# Optimizers

Optimization is a critical step in the deep learning process as it determines how well the network will learn from the training data.
SuperGradients supports out-of-the-box pytorch optimizers(
[SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD), 
[Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam), 
[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW),
[RMS_PROP](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop)), but also 
[RMSpropTF](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) and 
[Lamb](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py).

## Set the optimizer in the code
Optimizers should be part of the training parameters.


```py
from super_gradients import Trainer

trainer = Trainer(...)

trainer.train(
    training_params={"optimizer": "Adam", "optimizer_params": {"eps": 1e-3}, ...}, 
    ...
)
```

**Note**:
The `optimizer_params` is a dictionary of all the optimizer parameters you want to set. It can be any argument defined in the optimizer `__init__` method , except for `params` because this argument corresponds to the model to optimize and is automatically provided by the Trainer.


## Set the optimizer in the recipes
When working with recipes, you need to modify the [recipes/training_hyperparams](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes/training_hyperparams) as below:

```yaml
# recipes/training_hyperparams/my_training_recipe.yaml

...

optimizer: Adam
optimizer_params:
  eps: 1e-3
```


## Use Custom Optimizers
If your own optimizer is not natively supported by SuperGradients, you can always register it!

```py
from super_gradients.common.registry.registry import register_optimizer

@register_optimizer()
class CustomOptimizer:
    def __init__(
            self,
            params, # This arg is the only required regardless of your optimizer, the rest depends on your optimizer. 
            alpha: float, 
            betas: float
    ):
        defaults = dict(alpha=alpha, betas=betas)
        super(CustomOptimizer, self).__init__(params, defaults)

    ...
```

And then update your training hyperparameters:

```yaml
# my_training_hyperparams.yaml

...

optimizer: CustomOptimizer
optimizer_params:
  alpha: 1e-3
  betas: 1e-3
```

## Customize learning rate for different model blocks 
You can define the learning rate to use on each section of your model by working with `initialize_param_groups` and `update_param_groups`.
- `initialize_param_groups` defines the groups, and the learning rate to use for each group. It is called on instantiation.
- `update_param_groups` updates the learning rate for each group. It is called by LR callbacks (such as `LRCallbackBase`) during the training. 

If your model (i.e. any `torch.nn.Module`) is lacking these methods, the same learning rate will be applied to every block.
But if you implement them, it will be taken into account by the Trainer just like with any other SuperGradients model.

#### Example
Assuming that you have your own custom model and that you want work with a different learning rate on the backbone.

You first need to implement the `initialize_param_groups` and `update_param_groups` accordingly.


```py
import torch
from super_gradients.common.registry.registry import register_model


@register_model() # Required if working with recipe  
class MyModel(torch.nn.Module):

    ...

    def initialize_param_groups(self, lr: float, training_params) -> list:
        # OPTIMIZE BACKBONE USING CUSTOM LR
        backbone_params = {
            "named_params": self.backbone.named_parameters(),
            "lr": lr * training_params['multiply_backbone_lr'] # You can use any parameter, just make sure to define it when you set up training_params
        }

        # OPTIMIZE MAIN ARCHITECTURE LAYERS
        decoder_named_params = list(self.decoder.named_parameters())
        aux_head_named_parameters = list(self.aux_head.named_parameters())
        layers_params = {
            "named_params": decoder_named_params + aux_head_named_parameters,
            "lr": lr  
        }
        param_groups = [backbone_params, layers_params]
        return param_groups

    
    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params, total_batch: int) -> list:
        """
        Update the params_groups defined in initialize_param_groups
        """
        param_groups[0]["lr"] = lr * training_params['multiply_backbone_lr']

        param_groups[1]["lr"] = lr

        return param_groups
```
*Note: If working with recipe, don't forget to [register your model](configuration_files.md#registering-a-new-object).*


Now you just need to set a value for `multiply_backbone_lr` in the training recipe.
```yaml
# my_training_hyperparams.yaml

...

multiply_backbone_lr: 10 # This is used in our implementation of initialize_param_groups/update_param_groups
optimizer: OptimizerName # Any optimizer as described in the previous sections
optimizer_params: {} # Any parameter for the optimizer you chose
```
