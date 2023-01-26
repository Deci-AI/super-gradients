


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
You can stop the training after a few cycles. 

The recipe you have just used is a configuration file containing everything SG needs to know in order to train
Resnet18 on Cifar10. The actual YAML file is located in `src/super_gradients/recipes/cifar10_resnet.yaml`. In the same `recipes` library you can find many more
configuration files defining different models, datasets, and training hyper-parameters.

Try changing the `initial_lr` parameter in the file `src/super_gradients/recipes/training_hyperparams/cifar10_resnet_train_params.yaml` and launch this scrip again. 
You will see a different result now. This is because the parameters from `cifar10_resnet_train_params.yaml` are used in `cifar10_resnet.yaml`
(we will discuss thin in the next section). 

## Hydra (link to Hydra’s docs)
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

#### Command-Line Overrides
When running with Hydra, you can override or even add configuration from the command line. These override will apply to the specific run only.
```commandline
python train_from_recipe.py --config-name=cifar10_resnet training_hyperparams.initial_lr=0.02 experiment_name=test_lr_002
```
In the example above, the same script we launched earlier is used, but this time it will run with a different experiment name and a different 
initial learning-rate. This feature is extremely usefully when experimenting with different hyper-parameters.

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

The more advanced resolvers, are for 


## Recipes library structure

## Registering a new object 
with @register_x and using it in YAMLs
