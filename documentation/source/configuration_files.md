# Configuration Files and Recipes
SuperGradients supports [YAML](https://en.wikipedia.org/wiki/YAML) formatted configuration files. These files can contain training hyper-parameters,
architecture parameters, datasets parameters and any other parameters required by the training process.
These parameters will be consumed as dictionaries or as function arguments by different parts of SuperGradients.

> You can use SuperGradients without using any configuration files, look into the examples directory to see how.

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
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="recipes", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(cfg.experiment_name)
```

The `@hydra.main` decorator is looking for YAML files in the `super_gradients.recipes` according to the name of the configuration file provided 
in the first arg of the command line. 

In the experiment directory a `.hydra` subdirectory will be created. The configuration files related to this run will be saved by hydra to that subdirectory.  


Two Hydra features worth mentioning are _Command-Line Overrides_ and _YAML Composition_.


### Overriding Parameters with Command-Line

You'll often find the need to override or modify certain parameters for a specific run. 
Hydra offers a powerful feature that allows you to do this directly from the command line. 
This can be extremely useful for experimenting with different hyperparameters without changing the actual YAML files.

#### Syntax of Hydra Overrides

Here's how you can use Hydra overrides to modify parameters:

```shell
python -m super_gradients.train_from_recipe --config-name=<config-name> param1=<val1> path.to.param2=<val2> 
```

The arguments are referenced without the `--` prefix, 
and each parameter is referenced with its full path in the configuration tree, concatenated with a `.`.

#### Example of Command-Line Override

Consider the following example, where you want to run a training script on cifat10-resnet, 
but with a different experiment name and initial learning rate:

```shell
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet experiment_name=test_lr_002 training_hyperparams.initial_lr=0.02
```
