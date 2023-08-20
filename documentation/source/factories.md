# Working with Factories

Factories in SuperGradients provide a powerful and concise way to instantiate objects in your configuration files.

Prerequisites:
- [Training with Configuration Files](configuration_files.md)

In this tutorial, we'll cover how to use existing factories, register new ones, and briefly explore the implementation details.

## Utilizing Existing Factories

Let's start by looking at how existing factories can be utilized to define a sequence of transforms for augmenting a dataset.

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

In this example, SuperGradients will recognize the keys (`SegColorJitter`, `SegRandomFlip`, `SegRandomRescale`) 
which refer to SuperGradient's classes. They will be instantiated and passed to the Dataset constructor.

## Registering a Class

To use a new object from your configuration file, you need to define the mapping of the string to a type. 
This can be done using a registration functions.

Here's an example of how you can register a new model called `MyNet`:

```python
from super_gradients.common.registry import register_model

@register_model(name="MyNet")
class MyExampleNet(nn.Module):
    def __init__(self, num_classes: int):
        ....
```

This simple decorator maps the name "MyNet" to the type `MyExampleNet`. If your constructor includes required arguments,
you will be expected to provide them in your YAML file:

```yaml
architecture: 
    MyNet:
      num_classes: 8
```

Last step; make sure that you actually import the module including `MyExampleNet` into your script.
```python
from my_module import MyExampleNet # Importing the module is enough as it will trigger the register_model function

@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    Trainer.train_from_config(cfg)

def run():
    init_trainer()
    main()

if __name__ == "__main__":
    run()
```

## Under the Hood

Now, let's briefly look at how factories used within SuperGradients. If you want to explore this magic, you can look for the `@resolve_param` decorator in the code.

```python
class ImageNetDataset(torch_datasets.ImageFolder):
    
    @resolve_param("transforms", factory=TransformsFactory())
    def __init__(self, root: str, transforms: Union[list, dict] = [], *args, **kwargs):
        ...
        ...
```

The `@resolve_param` wraps functions and resolves a string or dictionary argument (in the example above "transforms") to an object. 
When `__init__(..)` is called, the function will receive an object, not a dictionary. 
The parameters under "transforms" in the YAML will be passed as arguments for instantiation.

## Supported Factory Tyoes
Each of the type 
```
    register_model
    register_kd_model
    register_detection_module
    register_metric
    register_loss
    register_dataloader
    register_callback
    register_transform
    register_dataset
    register_pre_launch_callback
    register_unet_backbone_stage
    register_unet_up_block
    register_target_generator
    register_lr_scheduler
    register_lr_warmup
    register_sg_logger
    register_collate_function
    register_sampler
    register_optimizer
    register_processing
```
