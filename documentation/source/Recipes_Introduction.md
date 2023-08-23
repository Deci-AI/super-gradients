# Introduction to Training Recipes

Recipes aim at providing a simple interface to easily reproduce trainings.

**Prerequisites**
- [Introduction to Configuration Files](configuration_files.md)


## Training from a Recipe

As explained in our [introduction to configuration files](configuration_files.md), SuperGradients uses the `hydra` 
library combined with `.yaml` recipes to allow you to easily customize the parameters.

The basic syntax to train a model from a recipe is a follows
```bash
python -m super_gradients.train_from_recipe --config-name=<config-name>
```
With `<config-name>` corresponding to the name of the recipe.

You can find all of the pre-defined recipes in [super_gradients/recipes](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes).
Recipe usually contain information about their performance, as well as the command to execute them in the header.

### Examples
- Training of Resnet18 on Cifar10: [super_gradients/recipes/cifar10_resnet.yaml](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/cifar10_resnet.yaml) 
```bash
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet
```

- Training of YoloX Small on COCO 2017 (8 GPUs): [super_gradients/recipes/coco2017_yolox](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/coco2017_yolox.yaml) 
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_s dataset_params.data_dir=/home/coco2017
```


## Customize Training
You may often need to modify certain parameters within a recipe and there 2 approaches for this: 
1. Using hydra overrides. 
2. Modifying the recipe.


### 1. Hydra Overrides

Hydra overrides allow you to change parameters directly from the command line.
This approach is ideal when want to quickly experiment changing a couple of parameters.  

Here's the general syntax:

```bash
python -m super_gradients.train_from_recipe --config-name=<config-name> param1=<val1> path.to.param2=<val2> 
```

- **Parameters** - Listed without the `--` prefix.
- **Full Path** - Use the entire path in the configuration tree, with each level separated by a `.`.


#### Example
Suppose your recipe looks like this:
```yaml
training_hyperparams:
  max_epochs: 250
  initial_lr: 0.1
  ...

dataset_params:
  data_dir: /local/mydataset
  ...

... # Many other parameters
```

Changing Epochs or Learning Rate
```bash
python -m super_gradients.train_from_recipe --config-name=<config-name> training_hyperparams.max_epochs=250 training_hyperparams.initial_lr=0.03
```

Changing the Dataset Path
```bash
python -m super_gradients.train_from_recipe --config-name=<config-name> dataset_params.data_dir=<path-to-dataset>
```

> Note: Parameter names may differ between recipes, so please check the specific recipe to ensure you're using the correct names.


### 2. Modifying the Recipe
If you are working on a cloned version of SuperGradients (`git clone ...`)
then you can directly modify existing recipes. 

If you installed SuperGradients with pip, then you won't have the ability to modify predefined recipes.
Instead, you should create your own recipe in your project, but you will still have the ability to build it on top of predefined recipes from SuperGradients.

We explain all of this in a [following tutorial](Recipes_Custom.md), but we strongly recommend you to 
first finish this tutorial, as it includes information required to fully understand how it works.


## Recipe Structure
If you brows the YAML files in the `recipes` directory you will see some file containing the saved-key `defaults` at the beginning of the file.

Here's an example of what this looks like:

```yaml
defaults:
  - training_hyperparams: cifar10_resnet_train_params
  - dataset_params: cifar10_dataset_params
  - arch_params: resnet18_cifar_arch_params
  - checkpoint_params: default_checkpoint_params
  - _self_

...
```

- **Defaults** - The `defaults` section leverages OmegaConf syntax to allow using other recipes as a base. 
- **Referencing Parameters** - You can reference parameters within the YAML file according to their origin. For example, `training_hyperparams.initial_lr` refers to the `initial_lr` parameter from the `cifar10_resnet_train_params.yaml` file.
- **Recipe Parameters** - `_self_` here means that the recipe itself will be used to override the defaults.

This last points means that if you have the following
```yaml
# training_hyperparams/default_training_recipe.yaml
initial_lr: 0.3
```
Either you use `_self_` last in the `defaults`
```yaml
# main_recipe.yaml
defaults:
  - training_hyperparams: default_training_recipe
  - _self_
    
training_hyperparams:
  initial_lr: 0.001 # 0.3 will be overriden by this [VALUE=0.001]
...
```
Or you use it first
```yaml
# main_recipe.yaml
defaults:
  - _self_
  - training_hyperparams: default_training_recipe
    
training_hyperparams:
  initial_lr: 0.001 # 0.001 will be overriden by the default [VALUE=0.03]
...
```

### Organizing Your Recipe Folder

Your recipe folder should have a specific structure to match this composition:

```
├─ cifar10_resnet.yaml
├─ ...
├─training_hyperparams
│   ├─ cifar10_resnet_train_params.yaml
│   └─ ...
├─dataset_params
│   ├─ cifar10_dataset_params.yaml
│   └─ ...
├─arch_params
│   ├─ resnet18_cifar_arch_params.yaml
│   └─ ...
└─checkpoint_params
    ├─ default_checkpoint_params.yaml
    └─ ...
```

You're not restricted to this structure, but following it ensures compatibility with SuperGradients' expectations.

### Override order
Something very important to remember about the `defaults` is the priority of override.



## Conclusion

This tutorial has introduced you to the world of training recipes within SuperGradients. Specifically, you've learned:
- **How to Train Models**: Utilizing `.yaml` recipes to effortlessly train and customize models.
- **Ways to Customize Training**: Tailoring your training through hydra overrides or direct modifications to the recipes.
- **Understanding Recipe Structure**: Grasping the organization and conventions that help you align with SuperGradients' expectations.

We've laid the groundwork for understanding how recipes enable flexible and reproducible training.

**Next Step**: In the [next tutorial](Recipes_Recipes_Factories.md), we'll explore factories in SuperGradients, revealing how they work with recipes to dynamically instantiate objects. It's a critical step in leveraging the full power of SuperGradients for your unique needs.

[Proceed to the next tutorial on Factories](Recipes_Recipes_Factories.md)