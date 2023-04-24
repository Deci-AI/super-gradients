# Configuration Files and Recipes
SuperGradients supports [YAML](https://en.wikipedia.org/wiki/YAML) formatted configuration files. These files can contain training hyper-parameters,
architecture parameters, datasets parameters and any other parameters required by the training process.
These parameters will be consumed as dictionaries or as function arguments by different parts of SuperGradients.

>You can use SuperGradients without using any configuration files, look into the examples directory to see how.

SuperGradients was designed to expose as many parameters as possible to allow outside configuration without writing a single line of code.
You can control the learning-rate, the weight-decay or even the loss function and metric used in the training, but moreover, you can even control 
which block-type or activation function to use in your model. You can learn about and define all of these parameters from the configuration files. 

Here is an example YAML file (training hyper-parameters in this case):
```yaml
defaults:
  - default_train_params

max_epochs: 250

lr_updates:
  _target_: numpy.arange
  start: 100
  stop: 250
  step: 50

lr_decay_factor: 0.1
lr_mode: step
lr_warmup_epochs: 0
initial_lr: 0.1
loss: cross_entropy
optimizer: SGD
criterion_params: {}

optimizer_params:
  weight_decay: 1e-4
  momentum: 0.9

```

## Why using configuration files
Using configuration file might seem too complicated or redundant at first. But, after a short training, you will find it extremely convenient and useful. 

Configuration file can help you manage your assets, such as datasets, models and training recipes. Keeping your code files as clean of parameters as possible,
allows you to have all of your configuration in one place and reuse the same code to define different objects.
In the following example, we define a training set and a validation set of Imagenet. both use the same piece of code
with different configurations:
```yaml
train_dataset_params:
  root: /data/Imagenet/train
  transforms:
    - RandomResizedCropAndInterpolation:
        size: 224
        interpolation: default
    - RandomHorizontalFlip
    - ToTensor
    - Normalize:
        mean: ${dataset_params.img_mean}
        std: ${dataset_params.img_std}

val_dataset_params:
  root: /data/Imagenet/val
  transforms:
    - Resize:
        size: 256
    - CenterCrop:
        size: 224
    - ToTensor
    - Normalize:
        mean: ${dataset_params.img_mean}
        std: ${dataset_params.img_std}
```

Configuration file can also help you track the exact settings used for each one of your experiments, tweak and tune these settings, and share them with others.
Concentrating all of these configuration parameters in one place, gives you great visibility and control of your experiments. 

## How to use configuration files
So, if you got so far, we have probably manged to convince you that configuration files are awsome and powerful tools - welcome aboard!

YAML is a human-readable data-serialization language. It is commonly used for configuration files and in applications where data is being 
stored or transmitted ([Wikipedia](https://en.wikipedia.org/wiki/YAML)). We parse each file into dictionaries, lists, and objects, and pass them to the code
either as a recursive dictionary or as function arguments. 

Let's try running a training session from a configuration file.
 
```shell
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet
```
You can stop the training after a few cycles. 

The recipe you have just used is a configuration file containing everything SG needs to know in order to train
Resnet18 on Cifar10. The actual YAML file is located in `src/super_gradients/recipes/cifar10_resnet.yaml`. In the same `recipes` library you can find many more
configuration files defining different models, datasets, and training hyper-parameters.

Try changing the `initial_lr` parameter in the file `src/super_gradients/recipes/training_hyperparams/cifar10_resnet_train_params.yaml` and launch this scrip again. 
You will see a different result now. This is because the parameters from `cifar10_resnet_train_params.yaml` are used in `cifar10_resnet.yaml`
(we will discuss thin in the next section). 

Two more useful functionalities are 
```commandline
python -m super_gradients.resume_experiment --experiment_name=cifar10_resnet
```

that will resume the experiment from the last checkpoint, and

```commandline
python -m super_gradients.evaluate_from_recipe --config-name=cifar10_resnet
```
that will run only the evaluation part of the recipe (without any training iterations)



## Hydra
Hydra is an open-source Python framework that provides us with many useful functionalities for YAML management. You can learn about Hydra 
[here](https://hydra.cc/docs/intro). We use Hydra to load YAML files and convert them into dictionaries, while 
instantiating the objects referenced in the YAML.
You can see this in the code:
```python
@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    Trainer.train_from_config(cfg)

def run():
    init_trainer()
    main()

if __name__ == "__main__":
    run()
```
The `@hydra.main` decorator is looking for YAML files in the `super_gradients.recipes` according to the name of the configuration file provided 
in the first arg of the command line. 

In the experiment directory a `.hydra` subdirectory will be created. The configuration files related to this run will be saved by hydra to that subdirectory.  

--------
Two Hydra features worth mentioning are _YAML Composition_ and _Command-Line Overrides_.

#### YAML Composition
If you brows the YAML files in the `recipes` directory you will see some file containing the saved-key `defaults:` at the beginning of the file.
```yaml
defaults:
  - training_hyperparams: cifar10_resnet_train_params
  - dataset_params: cifar10_dataset_params
  - arch_params: resnet18_cifar_arch_params
  - checkpoint_params: default_checkpoint_params
  - _self_

```
The YAML file containing this header will inherit the configuration of the above files. So when building a training recipe, one can structure
the configurations into a few files (for training hyper-params, dataset params, architecture params ect.) and Hydra will conveniently aggregate them all 
into a single dictionary. 

The parameters will be referenced inside the YAML according to their origin. i.e. in the example above we can reference `training_hyperparams.initial_lr` 
(initial_lr parameter from the cifar10_resnet_train_params.yaml file)

The aggregated configuration file will be saved in the `.hydra` subdirectory.

#### Command-Line Overrides
When running with Hydra, you can override or even add configuration from the command line. These override will apply to the specific run only.
```shell
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet training_hyperparams.initial_lr=0.02 experiment_name=test_lr_002
```
In the example above, the same script we launched earlier is used, but this time it will run with a different experiment name and a different 
initial learning-rate. This feature is extremely usefully when experimenting with different hyper-parameters.
Note that the arguments are referenced without the `--` prefix and that each parameter is referenced with its full path in the 
configuration tree, concatenated with a `.`.

## Resolvers 
Resolvers are converting the strings from the YAML file into Python objects or values. The most basic resolvers are the Hydra native resolvers.
Here are a few simple examples:
```yaml
a: 1
b: 2
c: 3
a_plus_b: "${add: ${a},${b}}"
a_plus_b_plus_c: "${add: ${a}, ${b}, ${c}}"
                 
my_list: [10, 20, 30, 40, 50]
third_of_list: "${getitem: ${my_list}, 2}"
first_of_list: "${first: ${my_list}}"
last_of_list: "${last: ${my_list}}"
```

The more advanced resolvers will instantiate objects. In the following example we define a few transforms that 
will be used to augment a dataset.
```yaml
train_dataset_params:
  transforms:
    # for more options see common.factories.transforms_factory.py
    - SegColorJitter:
        brightness: 0.1
        contrast: 0.1
        saturation: 0.1

    - SegRandomFlip:
        prob: 0.5

    - SegRandomRescale:
        scales: [ 0.4, 1.6 ]
```
Each one of the keys (`SegColorJitter`, `SegRandomFlip`, `SegRandomRescale`) is mapped to a type, and the configuration parameters under that key will be passed
to the type constructor by name (as key word arguments).

If you want to see where this magic is happening, you can look for the `@resolve_param` decorator in the code 

```python
class ImageNetDataset(torch_datasets.ImageFolder):
    
    @resolve_param("transforms", factory=TransformsFactory())
    def __init__(self, root: str, transforms: Union[list, dict] = [], *args, **kwargs):
        ...
        ...
```

The `@resolve_param` wraps functions and resolves a string or a dictionary argument (in the example above "transforms") to an object. 
To do so, it uses a factory object that maps a string or a dictionary to a type. when `__init__(..)` will be called, the function will receive 
an object, and not a dictionary. The parameters under "transforms" in the YAML will be passed as
arguments for instantiation the objects. We will  learn how to add a new type of object into these mappings in the next sections. 

## Registering a new object 
To use a new object from your configuration file, you need to define the mapping of the string to a type.
This is done using one of the many registration function supported by SG.
```python
register_model
register_detection_module
register_metric
register_loss
register_dataloader
register_callback
register_transform
register_dataset
```

These decorator functions can be imported and used as follows:

```python
from super_gradients.common.registry import register_model

@register_model(name="MyNet")
class MyExampleNet(nn.Module):
    def __init__(self, num_classes: int):
        ....
```

This simple decorator, maps the name "MyNet" to the type `MyExampleNet`. Note that if your constructor
include required arguments, you will be expected to provide them when using this string

```yaml
...
architecture: 
    MyNet:
      num_classes: 8
...

```

## Required Hyper-Parameters
Most parameters can be defined by default when including `default_train_params` in you `defaults`.
However, the following hyper-parameters are required to launch a training run:
```yaml
train_dataloader: 
val_dataloader: 
architecture: 
training_hyperparams:
  initial_lr: 
  loss:
experiment_name:
  
multi_gpu: # When training with multi GPU
num_gpus: # When training with multi GPU

# THE FOLLOWING PARAMS ARE DIRECTLY USED BY HYDRA
hydra:
  run:
    # Set the output directory (i.e. where .hydra folder that logs all the input params will be generated)
    dir: ${hydra_output_dir:${ckpt_root_dir}, ${experiment_name}}
```
> Other parameters may also be required, depending on the specific model, dataset, loss function ect. 
> Follow the error message in case you experiment did not launce properly.  

## Recipes library structure
The `super_gradients/recipes` include the following subdirectories:
> - arch_params - containing configuration files for instantiating different models
> - checkpoint_params - containing configuration files that define the loaded and saved checkpoints parameters for the training
> - conversion_params - containing configuration files for the model conversion scripts (for deployment)
> - dataset_params - containing configuration files for instantiating different datasets and dataloaders
> - training_hyperparams - containing configuration files holding hyper-parameters for specific recipes

These configuration files will be available for use both in the installed version and in the development version of SG.
