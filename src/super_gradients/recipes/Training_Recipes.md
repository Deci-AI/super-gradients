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



### List of recipes

All the commands to launch the recipes described [here](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes) are listed below.
Please make to `dataset_params.data_dir=<path-to-dataset>` if you did not store the dataset in the path specified by the recipe (as showed in the example above).

**Classification**
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

**Detection**

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


**Segmentation**

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




## Using your own recipe

### Recipe Structure
SuperGradients expects a very strict recipe structure which you should match.

# File structure
```


```

Please make sure to follow the structure 


> TIP: The simplest way to get started is probably for you to copy an existing recipe, and to overwrite anything you want to do differently.

### Running a recipe
When working straight from an existing recipe, we could run `python -m super_gradients.train_from_recipe --config-name=...`

Now that we are working with our own recipe which are not in the SuperGradients recipe folder, 
we will need to write a short custom script to launch them.

This script is essentially similar to `train_from_recipe`, but you will need to set yourself the path to your recipes
`config_path="<config-path>"`

Here is an example (adapted from the [train_from_recipe script](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/train_from_recipe.py)).

```python
# The code below is the same as the basic `train_from_recipe.py` script
# See: https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/train_from_recipe.py
from omegaconf import DictConfig
import hydra

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

The next step is 
