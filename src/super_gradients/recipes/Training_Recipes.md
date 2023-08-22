## Training Recipes

We defined recipes to ensure that anyone can reproduce our results in the most simple way.

Prerequisites:
- [Training with Configuration Files](configuration_files.md)


## Quickstart

The recipes are defined in `.yaml` format. SuperGradients use the hydra library to allow you to easily customize the parameters.

The basic syntax is as follows
```bash
python -m super_gradients.train_from_recipe --config-name=<config-name>
```

Example with **Cifar10** 
```bash
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet
```

> Please note that in many cases, you will need to 

### How to modify parameters
You will often need to override some parameters of the recipe, and there are 2 main approaches; 
modifying the recipe, or using hydra overrides.

We will discuss the first approach later, while explaining more generally how to write a recipe, 
so let's focus on the hydra overrides for now.

To syntax of hydra overrides is the following
```bash
python -m super_gradients.train_from_recipe --config-name=<config-name> param1=<val1> path.to.param2=<val2> 
```

The arguments are referenced without the `--` prefix, 
and each parameter is referenced with its full path in the configuration tree, concatenated with a `.`.

#### Example
Let's imagine we have a recipe with the following structure
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

**You want change the number of epoch or the learning rate.**
```bash
python -m super_gradients.train_from_recipe --config-name=<config-name> training_hyperparams.max_epochs=250 training_hyperparams.initial_lr=0.03
```

**You want to use an existing recipe which was defined with a dataset path different to yours**
```bash
python -m super_gradients.train_from_recipe --config-name=<config-name> dataset_params.data_dir=<path-to-dataset>
```

> Note: The name of some parameters may differ based on the recipe. 
> You should have a look at the recipe you want to use, and then make sure you are writing the write parameter name.


## Predefined Recipes

You can find all of our recipes [here](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes).
You will find information about the performance of a recipe as well as the command to execute it in the header of its config file.

*Example: [Training of YoloX Small on Coco 2017](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/coco2017_yolox.yaml), using 8 GPU* 
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_s dataset_params.data_dir=/home/coco2017
```

All the commands to launch the recipes described [here](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes) are listed below.
Please make to `dataset_params.data_dir=<path-to-dataset>` if you did not store the dataset in the path specified by the recipe (as showed in the example above).

### Classification
<details>
<summary>Cifar10</summary>

resnet
```bash
python -m super_gradients.train_from_recipe --config-name=cifar10_resnet +experiment_name=cifar10
```

</details>
<details>
<summary>ImageNet</summary>

efficientnet
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_efficientnet
```
mobilenetv2
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_mobilenetv2
```
mobilenetv3 small
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_mobilenetv3_small
```

mobilenetv3 large
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_mobilenetv3_large
```

regnetY200
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_regnetY architecture=regnetY200
```

regnetY400
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_regnetY architecture=regnetY400
```

regnetY600
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_regnetY architecture=regnetY600
```

regnetY800
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_regnetY architecture=regnetY800
```

repvgg
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_repvgg
```

resnet50
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_resnet50
```

resnet50_kd
```bash
python -m super_gradients.train_from_kd_recipe --config-name=imagenet_resnet50_kd
```

vit_base
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_vit_base
```

vit_large
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_vit_large
```
</details>

### Detection

<details>
<summary>Coco2017</summary>

ssd_lite_mobilenet_v2
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_ssd_lite_mobilenet_v2
```

yolox_n
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_n
```

yolox_t
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_t
```

yolox_s
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_s
```

yolox_m
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_m
```

yolox_l
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_l
```

yolox_x
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_x
```

</details>


### Segmentation

<details>
<summary>Cityscapes</summary>

DDRNet23
```bash
python -m super_gradients.train_from_recipe --config-name=cityscapes_ddrnet
```

DDRNet23-Slim
```bash
python -m super_gradients.train_from_recipe --config-name=cityscapes_ddrnet architecture=ddrnet_23_slim
```

RegSeg48
```bash
python -m super_gradients.train_from_recipe --config-name=cityscapes_regseg48
```

STDC1-Seg50
```bash
python -m super_gradients.train_from_recipe --config-name=cityscapes_stdc_seg50
```

STDC2-Seg50
```bash
python -m super_gradients.train_from_recipe --config-name=cityscapes_stdc_seg50 architecture=stdc2_seg
```

STDC1-Seg75
```bash
python -m super_gradients.train_from_recipe --config-name=cityscapes_stdc_seg75
```

STDC2-Seg75
```bash
python -m super_gradients.train_from_recipe --config-name=cityscapes_stdc_seg75 external_checkpoint_path=<stdc2-backbone-pretrained-path> architecture=stdc2_seg
```

</details>



## Recipe Structure
If you brows the YAML files in the `recipes` directory you will see some file containing the saved-key `defaults:` at the beginning of the file.

Here's an example of what this looks like:

```yaml
defaults:
  - training_hyperparams: cifar10_resnet_train_params
  - dataset_params: cifar10_dataset_params
  - arch_params: resnet18_cifar_arch_params
  - checkpoint_params: default_checkpoint_params

...
```

- **Defaults**: The `defaults` section leverages OmegaConf syntax to allow using other recipes as a base.
- **Referencing Parameters**: You can reference parameters within the YAML file according to their origin. For example, `training_hyperparams.initial_lr` refers to the `initial_lr` parameter from the `cifar10_resnet_train_params.yaml` file.


## Required Parameters
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
> - Other parameters may also be required, depending on the specific model, dataset, loss function ect. 
> - Follow the error message in case you experiment did not launce properly.  


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


### Training with custom recipe
When working straight from an existing recipe, we could run `python -m super_gradients.train_from_recipe --config-name=...`

Now that we are working with our own recipe which are not in the SuperGradients recipe folder, 
we will need to write a short custom script to launch them.

This script is essentially similar to `train_from_recipe`, but you will need to set yourself the path to your recipes
`config_path="<config-path>"`

Here is an example (adapted from the [train_from_recipe script](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/train_from_recipe.py)).

```python
# The code below is the same as the basic `train_from_recipe.py` script
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

### Customizing your training script

In some rare cases, you may want to combine using recipes with writing some custom code before launching the Training.
Here is an example on how recipes can be used, while keeping the flexibility of python code.

```python
import hydra
from omegaconf import DictConfig

from super_gradients import Trainer, init_trainer, setup_device
from super_gradients.training import dataloaders, models

@hydra.main(config_path="<config-path>", version_base="1.2") # TODO: overwrite `<config-path>`
def _main(cfg: DictConfig) -> None:
        setup_device(
            device=cfg.devuce,
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
        train_results = trainer.train(
            model=model,
            train_loader=train_dataloader,
            valid_loader=val_dataloader,
            training_params=cfg.training_hyperparams,
        )
        
        print(train_results)

def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()

if __name__ == "__main__":
    main()
```
