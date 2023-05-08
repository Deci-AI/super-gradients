
## Training Recipes

We defined recipes to ensure that anyone can reproduce our results in the most simple way.


**Setup**

To run recipes you first need to clone the super-gradients repository:
```bash
git clone https://github.com/Deci-AI/super-gradients
```

You then need to move to the root of the clone project (where you find "requirements.txt" and "setup.py") and install super-gradients:
```bash
pip install -e .
```

Finally, append super-gradients to the python path: (Replace "YOUR-LOCAL-PATH" with the path to the downloaded repo)
```bash
export PYTHONPATH=$PYTHONPATH:<YOUR-LOCAL-PATH>/super-gradients/
```


**How to run a recipe**

The recipes are defined in .yaml format and we use the hydra library to allow you to easily customize the parameters.
The basic basic syntax is as follow:
```bash
python -m super_gradients.train_from_recipe --config-name=<CONFIG-NAME> dataset_params.data_dir=<PATH-TO-DATASET>
```
*Note: this script needs to be launched from the root folder of super_gradients*
*Note: if you stored your dataset in the path specified by the recipe you can drop "dataset_params.data_dir=<PATH-TO-DATASET>".*

**Explore our recipes**

You can find all of our recipes [here](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes).
You will find information about the performance of a recipe as well as the command to execute it in the header of its config file.

*Example: [Training of YoloX Small on Coco 2017](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/coco2017_yolox.yaml), using 8 GPU* 
```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox architecture=yolox_s dataset_params.data_dir=/home/coco2017
```



**List of commands**

All the commands to launch the recipes described [here](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes) are listed below.
Please make to "dataset_params.data_dir=<PATH-TO-DATASET>" if you did not store the dataset in the path specified by the recipe (as showed in the example above).

**- Classification**
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

**- Detection**

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


**- Segmentation**

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
