# Training Recipes

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
Recipes usually contain information about their performance, as well as the command to execute them in the header.

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
You may often need to modify certain parameters within a recipe and there are 2 approaches for this: 
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
When browsing the YAML files in the `recipes` directory, you'll notice that some files contain the key `defaults` at the beginning of the file. 
Here's an example of what this looks like:

```yaml
defaults:
  - training_hyperparams: cifar10_resnet_train_params
  - dataset_params: cifar10_dataset_params
  - arch_params: resnet18_cifar_arch_params
  - checkpoint_params: default_checkpoint_params
  - _self_
  - variable_setup

architecture: resnet18

train_dataloader: cifar10_train # Optional, see comments below
val_dataloader: cifar10_val # Optional, see comments below

multi_gpu: Off
num_gpus: 1

experiment_suffix: ""
experiment_name: cifar10_${architecture}${experiment_suffix}
```

This is a _minimal_ example of the recipe file that contains all **mandatory** properties to train a model.

### Components of a Recipe

We need to introduce some terminology to ensure we stay on the same page throughout the rest of this document.

- **Defaults**: The `defaults` section is critical, and it leverages the OmegaConf syntax. It serves to reference other recipes, allowing you to create modular and reusable configurations.
- **Referencing Parameters**: This allows you to point to specific parameters in the YAML file according to where they originate. For example, `training_hyperparams.initial_lr` refers to the `initial_lr` parameter from the `cifar10_resnet_train_params.yaml` file.
- **Recipe Parameters - `_self_`**: The `_self_` keyword has a special role. It permits the current recipe to override the defaults. Its impact depends on its position in the `defaults` list.

A recipe consists of a several sections that are mandatory and required to exist in the recipe file. They are:

- **`training_hyperparams`** - This section contains the hyperparameters related to training regime, such as the learning rate, number of epochs, etc.
- **`dataset_params`** - This section contains the parameters related to the dataset and dataloaders for training and validation. Dataset transformations, batch size, etc. are defined here.
The `dataset_params` section is tightly coupled with the root parameters `train_dataloader` and `val_dataloader`.
Please note, that `train_dataloader` and `val_dataloader` are optional, not mandatory parameters in a broad sense.
They are used in conjunction to instantiate the dataloaders for training and validation and exists mostly for convenience purposes in SG-provided recipes.
For **external** datasets we suggest read the [Using Custom Datasets](https://docs.deci.ai/super-gradients/documentation/source/Data.html#using-custom-datasets) section of Datasets documentation page for additional information.
- **`arch_params`** - This section contains the parameters related to the model architecture. The `arch_params` section goes hand-in-hand with the `architecture` parameter, which is root property of the recipe. 
If `architecture`defines the specific model architecture, then `arch_params` defines the parameters for that architecture.
- **`checkpoint_params`** - This section contains  the parameters related to checkpoints. 
It contains settings for loading checkpoint weights for transfer learning, controlling use of pretrained weights and more.
See [default_checkpoint_params](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/checkpoint_params/default_checkpoint_params.yaml) for an example of what parameters are supported.
- **`variable_setup`**: This section is required to enable use of shortcuts for most commonly used overrides which is covered in the [next](#Command-Line Override Shortcuts) section. Please note it `variable_setup` **must be the last item** in the defaults list.

### Understanding Override Order

> 🚨 **Warning**: The order of items in the `defaults` section is significant! The overwrite priority follows the list order, meaning that a config defined higher in the list can be overwritten by one defined lower in the list. This is a vital aspect to be aware of when constructing recipes. For a more detailed explanation, please refer to the [official documentation](https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/#composition-order-of-primary-config).

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

### Command-Line Override Shortcuts

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
Please note it `variable_setup` **must be the last item** in the defaults list.

## Conclusion

This tutorial has introduced you to the world of training recipes within SuperGradients. Specifically, you've learned:
- **How to Train Models**: Utilizing `.yaml` recipes to effortlessly train and customize models.
- **Ways to Customize Training**: Tailoring your training through hydra overrides or direct modifications to the recipes.
- **Understanding Recipe Structure**: Grasping the organization and conventions that help you align with SuperGradients' expectations.

We've laid the groundwork for understanding how recipes enable flexible and reproducible training.

**Next Step**: In the [next tutorial](Recipes_Factories.md), we'll explore factories in SuperGradients, 
revealing how they work with recipes to dynamically instantiate objects. It's a critical step in leveraging the 
full power of SuperGradients for your unique needs.
