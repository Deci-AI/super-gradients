## Training on Custom Recipes


Prerequisites:
- [Introduction to Configuration Files](configuration_files.md)
- [Introduction to Training Recipes](Recipes_Training.md)
- [Working with Factories](Recipes_Factories.md)


In this section, we will assume that you want to build you own recipe, and to train a model based on that recipe.

We will cover 2 different approaches in writing your recipe.
1. **SuperGradients Format** - you stick to the format used in SuperGradients.
2. **Custom Format** - you organize recipes the way you want.


### 1. SuperGradient Format 
This approach is most appropriate when you want to quickly get started. 

Since you will be following all SuperGradients convention when building the recipe, 
you won't have to worry about working with hydra to instantiate your objects and to launch a training; SuperGradients already provides a script that will do it for you.

**How to get started?**

1. We recommend that you would go through the [pre-defined recipes](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/)
and chose the one which seems most similar to your use case. Make sure it covers the same task as you.
2. Copy it to a folder that will be exclusively meant for recipes, inside your project.
3. Override the required parameters to fit your needs. Make sure to keep the same structure. Think about [registering custom objects](Recipes_Factories.md) if you need. 
4. Copy [train_from_recipe script](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/train_from_recipe.py) to your project (see below), but think to override `<config-path>` with the path to your recipe folder.
 

```python
# The code below is the same as the `train_from_recipe.py` script
# See: https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/train_from_recipe.py
import hydra
from omegaconf import DictConfig
from super_gradients import Trainer, init_trainer

@hydra.main(config_path="<config-path>", version_base="1.2") # TODO: overwrite `<config-path>`
def _main(cfg: DictConfig) -> None:
    Trainer.train_from_config(cfg)

def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()

if __name__ == "__main__":
    main()
```


### 2. Customizing Recipe Format

With this approach, you will have much more freedom in the way you organize your recipe but this will come at the cost of writing code!
This is mainly recommended for specific use-cases which are not properly covered with the previous approach.

Despite not being required with this approach, we strongly recommend for you to use the same format as in 
SuperGradients as it would allow you to build on top of pre-defined recipes.


**What are the recipe format constraints here ?**

With this approach, you will still need to follow certain conventions
- `training_hyperparams` should include the same required fields as with the previous approach. You can find the list [here](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml).
- The config passed to `dataloaders.get` should still be compatible to dataset/dataloader you want to load. 

Basically, the format constraints that you will face with this approach are the same as these that you would face when working exclusively with python.


**How to launch a training ?**

Similarly to the previous approach, you will need a script that will launch the training.
The difference being that here you won't be using `Trainer.train_from_config`. Instead, you will to isntantiate all the required objects in your script.

Here is an example of how such a script could look like:
```python
import hydra
from omegaconf import DictConfig

from super_gradients import Trainer, init_trainer, setup_device
from super_gradients.training import dataloaders, models

@hydra.main(config_path="<config-path>", version_base="1.2") # TODO: overwrite `<config-path>`
def _main(cfg: DictConfig) -> None:
    setup_device(
        device=cfg.device,
        multi_gpu=cfg.multi_gpu,
        num_gpus=cfg.num_gpus,
    )

    # INSTANTIATE ALL OBJECTS IN CFG
    cfg = hydra.utils.instantiate(cfg)

    trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir)

    # BUILD NETWORK
    model = models.get(
        model_name=cfg.architecture,
        num_classes=cfg.arch_params.num_classes,
        arch_params=cfg.arch_params,
        strict_load=cfg.checkpoint_params.strict_load,
        pretrained_weights=cfg.checkpoint_params.pretrained_weights,
        checkpoint_path=cfg.checkpoint_params.checkpoint_path,
        load_backbone=cfg.checkpoint_params.load_backbone,
    )

    # INSTANTIATE DATA LOADERS
    train_dataloader = dataloaders.get(
        name=cfg.train_dataloader,
        dataset_params=cfg.dataset_params.train_dataset_params,
        dataloader_params=cfg.dataset_params.train_dataloader_params,
    )

    val_dataloader = dataloaders.get(
        name=cfg.val_dataloader,
        dataset_params=cfg.dataset_params.val_dataset_params,
        dataloader_params=cfg.dataset_params.val_dataloader_params,
    )

    # TRAIN
    results = trainer.train(
        model=model,
        train_loader=train_dataloader,
        valid_loader=val_dataloader,
        training_params=cfg.training_hyperparams,
        additional_configs_to_log={},
    )
    print(results)


def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()

if __name__ == "__main__":
    main()
```


## Tips

### Building on top of SuperGradients Recipes
By default, `defaults` only works with recipes that are defined in the same recipe directory, but this can be extended to other directories.

In our case, this comes handy when you want to build on top of recipes that were implemented in SuperGradients.

#### Example

Using `default_train_params` defined in [super_gradients/recipes/training_hyperparams/default_train_params.yaml](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml)

```yaml
defaults:
  - training_hyperparams: default_train_params 

hydra:
  searchpath:
    - pkg://super_gradients.recipes
    
... # Continue with your recipe
```
