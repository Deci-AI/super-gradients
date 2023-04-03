# DeciYolo Quickstart
<div>
<img src="images/deciyolo_frontier.png" width="750">
</div>
Deci's research team developed a new deep learning architecture that competes with YOLOv8 using their efficient AutoNAC algorithm. They incorporated quantization-aware RepVGG blocks into the model architecture to ensure compatibility with Post-Training Quantization, making it more flexible and usable for different hardware configurations.


In this tutorial, we will go over the basic functionalities of DeciYolo very briefly.



## Instantiate a DeciYolo Model

```python
from super_gradients.training import models
from super_gradients.common.object_names import Models

net = models.get(Models.DECIYOLO_S, pretrained_weights="coco")
```

## Predict
```python
#TODO: ADD ONCE PREDICT IS IMPLEMENTED
```


## Train on RF100


Follow the setup instructions for RF100:
```
        - Follow the official instructions to download Roboflow100: https://github.com/roboflow/roboflow-100-benchmark?ref=roboflow-blog
            //!\\ To use this dataset, you must download the "coco" format, NOT the yolov5.

        - Your dataset should look like this:
            rf100
            ├── 4-fold-defect
            │      ├─ train
            │      │    ├─ 000000000001.jpg
            │      │    ├─ ...
            │      │    └─ _annotations.coco.json
            │      ├─ valid
            │      │    └─ ...
            │      └─ test
            │           └─ ...
            ├── abdomen-mri
            │      └─ ...
            └── ...

        - Install CoCo API: https://github.com/pdollar/coco/tree/master/PythonAPI
```
We first clone the SG repo, then use the repo's configuration files in our training examples.
We will use the ```src/super_gradients/recipes/roboflow_deciyolo_s.yaml```configuration to train the small variant of our DeciYolo, DeciYolo S.

So we navigate to our ```train_from_recipe``` script:
```commandline
cd <YOUR-LOCAL-PATH>/super_gradients/src/super_gradients/examples/train_from_recipe_example
```

Then to avoid collisions between our cloned and installed SG:
```commandline
export PYTHONPATH=<YOUR-LOCAL-PATH>/super_gradients/:<YOUR-LOCAL-PATH>/super_gradients/src
```

To launch training on one of the RF100 datasets, we pass it through the dataset_name argument:
```
python -m train_from_recipe --config-name=roboflow_deciyolo_s  dataset_name=<DATASET_NAME> dataset_params.data_dir=<PATH_TO_RF100_ROOT> ckpt_root_dir=<YOUR_CHECKPOINTS_ROOT_DIRECTORY>
```

Replace <DATASET_NAME> with any of the [RF100 datasets](https://github.com/roboflow/roboflow-100-benchmark/blob/8587f81ef282d529fe5707c0eede74fe91d472d0/metadata/datasets_stats.csv) that you wish to train on.
