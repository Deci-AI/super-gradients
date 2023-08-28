# Configuration Files and Recipes
SuperGradients supports [YAML](https://en.wikipedia.org/wiki/YAML) formatted configuration files. 
These files can contain training hyper-parameters, architecture parameters, datasets parameters and any 
other parameters required by the training process.

These parameters will be consumed as dictionaries or as function arguments by different parts of SuperGradients.

> These YAML files act like a cookbook for training models, which is why they are called **Recipes**.

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


> NOTE: You can use SuperGradients without using any configuration files, look into the examples directory to see how.

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
        mean: [0.485, 0.456, 0.406] # mean for normalization
        std: [0.229, 0.224, 0.225]  # std  for normalization

val_dataset_params:
  root: /data/Imagenet/val
  transforms:
    - Resize:
        size: 256
    - CenterCrop:
        size: 224
    - ToTensor
    - Normalize:
        mean: [0.485, 0.456, 0.406] # mean for normalization
        std: [0.229, 0.224, 0.225]  # std  for normalization
```

Configuration file can also help you track the exact settings used for each one of your experiments, tweak and tune these settings, and share them with others.
Concentrating all of these configuration parameters in one place, gives you great visibility and control of your experiments. 

## How to use configuration files
So, if you got so far, we have probably manged to convince you that configuration files are awsome and powerful tools - welcome aboard!

YAML is a human-readable data-serialization language. It is commonly used for configuration files and in applications where data is being 
stored or transmitted ([Wikipedia](https://en.wikipedia.org/wiki/YAML)). 
We parse each file into dictionaries, lists, and objects, and pass them to the code either as a recursive dictionary or as function arguments. 

Let's try running a training session from a configuration file.
 
```shell
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet
```
You can stop the training after a few cycles. 

The recipe you have just used is a configuration file containing everything SG needs to know in order to train
Resnet18 on Cifar10. The actual YAML file is located in `src/super_gradients/recipes/cifar10_resnet.yaml`. 
In the same `recipes` library you can find many more configuration files defining different models, datasets, 
and training hyper-parameters.


### Hydra
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

Two Hydra features worth mentioning are [Command-Line Overrides](https://hydra.cc/docs/advanced/override_grammar/basic/) 
and [YAML Composition](https://hydra.cc/docs/0.11/tutorial/composition/).
We strongly recommend you to have a look at both of these pages.

#### YAML Composition
If you browse the YAML files in the `recipes` directory you will see some file containing the saved-key `defaults:` at the beginning of the file.
```yaml
defaults:
  - training_hyperparams: cifar10_resnet_train_params
  - dataset_params: cifar10_dataset_params
  - arch_params: resnet18_cifar_arch_params
  - checkpoint_params: default_checkpoint_params
  - _self_
  - variable_setup

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

##### Command-Line Override Shortcuts

Although you can override any parameter from the command line, writing the full path of the parameter can be tedious.
For example, to change the learning rate one would have to write `training_hyperparams.initial_lr=0.02`. 
To change the batch size one would have to write
`dataset_params.train_dataloader_params.batch_size=128 dataset_params.val_dataloader_params.batch_size=128`.

To make it easier, we have defined a few shortcuts for the most common parameters that aims to reduce the amount of typing required:

* Learning rate: `lr=0.02` (same as `training_hyperparams.initial_lr=0.02`)
* Batch size: `bs=128` (same as `dataset_params.train_dataloader_params.batch_size=128 dataset_params.val_dataloader_params.batch_size=128`)
* Number of train epochs: `epochs=100` (same as `training_hyperparams.max_epochs=100`)
* Number of workers: `num_workers=4` (same as `dataset_params.train_dataloader_params.num_workers=4 dataset_params.val_dataloader_params.num_workers=4`)
* Resume training for a specific experiment: `resume=True` (same as `training_hyperparams.resume=True`)
* Enable or disable EMA: `ema=true` (same as `training_hyperparams.ema=true`)

To use these shortcuts, a `variable_setup` section should be a part of hydra defaults in the recipe file.
Please note it `variable_setup` **must be the last item** in the defaults list:

```yaml
defaults:
  - training_hyperparams: cifar10_resnet_train_params
  - dataset_params: cifar10_dataset_params
  - arch_params: resnet18_cifar_arch_params
  - checkpoint_params: default_checkpoint_params
  - _self_
  - variable_setup

...
```

More informaiton can be found in corresponding YAML file in the `recipes` directory:
https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/variable_setup.yaml

### Conclusion
This brief introduction has given you a glimpse into the functionality and importance of recipes within SuperGradients:
- **Recipes Overview**: Configuration files in YAML format that allow streamlined training and customization.
- **SuperGradients' Utilization**: Enhancing reproducibility, flexibility, and efficiency in defining models, datasets, and hyperparameters.
- **Introduction to training**: A simple demonstration of initiating a training session using a specific recipe.

**Next Step**: More details await in the [upcoming tutorials](Recipes_Training.md), where we'll explore more in-depth training from recipes, 
and the customization, structure, and deeper functionality of recipes within SuperGradients. 
