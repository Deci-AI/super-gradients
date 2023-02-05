# Optimizers

Optimization is a critical step in the deep learning process as it determines how well the network will learn from the training data.
SuperGradients supports out-of-the-box pytorch optimizers(
[SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD), 
[Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam) and 
[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW)
), but also 
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
