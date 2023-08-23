# Working with Factories

Factories in SuperGradients provide a powerful and concise way to instantiate objects in your configuration files.

Prerequisites:
- [Training with Configuration Files](configuration_files.md)
- [Introduction to Training Recipes](Recipes_Training.md)

In this tutorial, we'll cover how to use existing factories, register new ones, and briefly explore the implementation details.

## Using Existing Factories

If you had a look at the [recipes](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes), you may have noticed that many objects are defined directly in the recipes.

In the [Supervisely dataset recipe](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/dataset_params/supervisely_persons_dataset_params.yaml) you can see the following

```yaml
train_dataset_params:
  transforms:
    - SegColorJitter:
        brightness: 0.1
        contrast: 0.1
        saturation: 0.1
    - SegRandomFlip:
        prob: 0.5
    - SegRandomRescale:
        scales: [0.4, 1.6]
```
If you load the `.yaml` recipe as is into a python dictionary, you would get the following
```python
{
  "train_dataset_params": {
    "transforms": [
      {
        "SegColorJitter": {
          "brightness": 0.1,
          "contrast": 0.1,
          "saturation": 0.1
        }
      },
      {
        "SegRandomFlip": {
          "prob": 0.5
        }
      },
      {
        "SegRandomRescale": {
          "scales": [0.4, 1.6]
        }
      }
    ]
  }
}
```

This configuration alone is not very useful, as we need instances of the classes, not just their configurations.
So we would like to somehow instantiate these classes `SegColorJitter`, `SegRandomFlip` and `SegRandomRescale`.

Factories in SuperGradients come into play here! All these objects were registered beforehand in SuperGradients, 
so that when you write these names in the recipe, SuperGradients will detect and instantiate them for you.

## Registering a Class

As explained above, only registered objects can be instantiated. 
This registration consists of mapping the object name to the corresponding class type.

In the example above, the string `"SegColorJitter"` was mapped to the class `SegColorJitter`, and this is how SuperGradients knows how to convert the string defined in the recipe, into an object.

You can register the class using a name different from the actual class name. 
However, it's generally recommended to use the same name for consistency and clarity.

### Example

```python
from super_gradients.common.registry import register_transform

@register_transform(name="MyTransformName")
class MyTransform:
    def __init__(self, prob: float):
        ...
```
In this simple example, we register a new transform.
Note that here we registered (for the sake of the example) the class `MyTransform` to the name `MyTransformName` which is different. 
We strongly recommend to not do it, and to instead register a class with its own name.

Once you registered a class, you can use it in your recipe. Here, we will add this transform to the original recipe
```yaml
train_dataset_params:
  transforms:
    - SegColorJitter:
        brightness: 0.1
        contrast: 0.1
        saturation: 0.1
    - SegRandomFlip:
        prob: 0.5
    - SegRandomRescale:
        scales: [0.4, 1.6]
    - MyTransformName:  # We use the name used to register, which may be different from the name of the class
        prob: 0.7 
```

Final Step: Ensure that you import the module containing `MyTransformName` into your script. 
Doing so will trigger the registration function, allowing SuperGradients to recognize it.

Here is an example (adapted from the [train_from_recipe script](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/train_from_recipe.py)).

```python
from .my_module import MyTransform # Importing the module is enough as it will trigger the register_transform function

# The code below is the same as the basic `train_from_recipe.py` script
# See: https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/train_from_recipe.py
from omegaconf import DictConfig
import hydra

from super_gradients import Trainer, init_trainer


@hydra.main(config_path="recipes", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    Trainer.train_from_config(cfg)


def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()


if __name__ == "__main__":
    main()

```

## Under the Hood

Until now, we saw how to use existing Factories, and how to register new ones.
In some cases, you may want to create objects that would benefit from using the factories.

### Basic
The basic way to use factories as below.
```
from super_gradients.common.factories import TransformsFactory
factory = TransformsFactory()
my_transform = factory.get({'MyTransformName': {'prob':  0.7}})
```
You may recognize that the input passed to `factory.get` is actually the dictionary that we get after loading the recipe
(See [Utilizing Existing Factories](#utilizing-existing-factories))

### Recommended
Factories become even more powerful when used with the `@resolve_param` decorator. 
This feature allows functions to accept both instantiated objects and their dictionary representations. 
It means you can pass either the actual python object or a dictionary that describes it straight from the recipe.

```python
class ImageNetDataset(torch_datasets.ImageFolder):
    
    @resolve_param("transforms", factory=TransformsFactory())
    def __init__(self, root: str, transform: Transform):
        ...
```

Now, `ImageNetDataset` can be passed both an instance of `MyTransform`

```python
my_transform = MyTransform(prob=0.7)
ImageNetDataset(root=..., transform=my_transform)
```

And a dictionary representing the same object
```python
my_transform = {'MyTransformName': {'prob':  0.7}}
ImageNetDataset(root=..., transform=my_transform)
```

This second way of instantiating the dataset combines perfectly with the concept `.yaml` recipes.

**Difference with `register_transform`**
- `register_transform` is responsible to map a string to a class type.
- `@resolve_param("transform", factory=TransformsFactory())` is responsible to convert a config into an object, using the mapping created with `register_transform`. 

## Supported Factory Types
Until here, we focused on a single type of factory, `TransformsFactory`, 
associated with the registration decorator `register_transform`. 

SuperGradients supports a wide range of factories, used throughout the training process, 
each with its own registering decorator.
 
SuperGradients offers various types of factories, and each is associated with a specific registration decorator.

``` python
from super_gradients.common.factories import (
    register_model,
    register_kd_model,
    register_detection_module,
    register_metric,
    register_loss,
    register_dataloader,
    register_callback,
    register_transform,
    register_dataset,
    register_pre_launch_callback,
    register_unet_backbone_stage,
    register_unet_up_block,
    register_target_generator,
    register_lr_scheduler,
    register_lr_warmup,
    register_sg_logger,
    register_collate_function,
    register_sampler,
    register_optimizer,
    register_processing,
)
```

### Conclusion

In this tutorial, we have delved into the realm of factories, encompassing:
- **Using Existing Factories**: How SuperGradients automatically instantiates objects defined in recipes.
- **Registering New Classes**: The method to map object names to corresponding class types, and how to integrate them in your recipes.
- **Under the Hood**: Insights into basic and recommended ways to use factories, as well as the variety of supported factory types within SuperGradients.

These insights provide essential understanding and practical techniques to work with factories, a core element in SuperGradients that bridges the gap between configuration and instantiation.

**Next Step**: Ready to craft your unique recipes? In the [next tutorial](Recipes_Custom.md), 
we'll guide you through building your own recipe and training a model based on that recipe. 
