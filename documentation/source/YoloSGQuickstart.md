# YoloSG Quickstart
<div>
<img src="images/yolo_sg_frontier.png" width="750">
</div>
We developed a new deep learning architecture that competes with YOLOv8 using their efficient AutoNAC algorithm. 
We incorporated quantization-aware RepVGG blocks into the model architecture to ensure compatibility with Post-Training Quantization, making it more flexible and usable for different hardware configurations.


In this tutorial, we will go briefly over the basic functionalities of YoloSG.



## Instantiate a YoloSG Model

```python
from super_gradients.training import models
from super_gradients.common.object_names import Models

net = models.get(Models.YoloSG_S, pretrained_weights="coco")
```

## Predict
```python
prediction = net.predict("https://www.aljazeera.com/wp-content/uploads/2022/12/2022-12-03T205130Z_851430040_UP1EIC31LXSAZ_RTRMADP_3_SOCCER-WORLDCUP-ARG-AUS-REPORT.jpg?w=770&resize=770%2C436&quality=80")
```
<div>
<img src="images/yolo_sg_qs_predict.png" width="750">
</div>

## Export to ONNX
```python
models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="yolo_sg_s.onnx")
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
We will use the ```roboflow_yolo_sg_s```configuration to train the small variant of our YoloSG, YoloSG-S.

To launch training on one of the RF100 datasets, we pass it through the dataset_name argument:
```
python -m super_gradients.train_from_recipe --config-name=roboflow_yolo_sg_s  dataset_name=<DATASET_NAME> dataset_params.data_dir=<PATH_TO_RF100_ROOT> ckpt_root_dir=<YOUR_CHECKPOINTS_ROOT_DIRECTORY>
```

Replace <DATASET_NAME> with any of the [RF100 datasets](https://github.com/roboflow/roboflow-100-benchmark/blob/8587f81ef282d529fe5707c0eede74fe91d472d0/metadata/datasets_stats.csv) that you wish to train on.
