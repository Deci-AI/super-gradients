


# Configuration files and Recipes
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

## Why to use configuration files
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
In your SG project, go to the examples directory and run the `train_from_recipe.py` script as shown below: 
```commandline
cd src/super_gradients/examples/train_from_recipe_example
python train_from_recipe.py --config-name=cifar10_resnet
```
you can stop the training after a few cycles. 

The recipe you have just used is a configuration file containing 

## Hydra (link to Hydra’s docs)

## Resolvers 


## Recipes library structure

## Registering a new object 
with @register_x and using it in YAMLs
