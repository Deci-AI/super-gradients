# Object Detection

Object detection is a core task in computer vision that allows to detect and classify bounding boxes in images. 
It's been gaining popularity and ubiquity extremely fast since the first breakthroughs in Deep Learning and advanced a wide range of companies, including the medical domain, surveillance, smart shopping, etc.
It comes as no surprise considering that it covers two basic needs in an end-to-end manner: to find all present objects and to assign a class to each one of them, 
while cleverly dealing with the background and its dominance over all other classes. 

Due to this, most recent research publications dedicated to object detection focus on a good trade-off between accuracy and speed.
In SuperGradients, we aim to collect such models and make them very convenient and accessible to you, so that you can try any one of them interchangeably.

## Implemented models

| Model                                    | Yaml                                                                                                                                           | Model class                                                                                                                                     |  Loss Class                                                                                                     | NMS Callback                                                                                                                                                                            |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
| [SSD](https://arxiv.org/abs/1512.02325) | [ssd_lite_mobilenetv2_arch_params](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/arch_params/ssd_lite_mobilenetv2_arch_params.yaml) | [SSDLiteMobileNetV2](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/ssd.py) | [SSDLoss](https://docs.deci.ai/super-gradients/docstring/training/losses/#training.losses.ssd_loss.SSDLoss) | [SSDPostPredictCallback](https://docs.deci.ai/super-gradients/docstring/training/utils/#training.utils.ssd_utils.SSDPostPredictCallback) |
| [YOLOX](https://arxiv.org/abs/2107.08430) | [yolox_s_arch_params](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/arch_params/yolox_s_arch_params.yaml) | [YoloX_S](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/yolox.py) | [YoloXFastDetectionLoss](https://docs.deci.ai/super-gradients/docstring/training/losses/#training.losses.yolox_loss.YoloXFastDetectionLoss) | [YoloPostPredictionCallback](https://docs.deci.ai/super-gradients/docstring/training/models/#training.models.detection_models.yolo_base.YoloPostPredictionCallback) |
| [PPYolo](https://arxiv.org/abs/2007.12099) | [ppyoloe_arch_params](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/arch_params/ppyoloe_arch_params.yaml) | [PPYoloE](https://docs.deci.ai/super-gradients/docstring/training/models/#training.models.detection_models.pp_yolo_e.pp_yolo_e.PPYoloE) | [PPYoloELoss](https://docs.deci.ai/super-gradients/docstring/training/losses/#training.losses.ppyolo_loss.PPYoloELoss) | [PPYoloEPostPredictionCallback](https://docs.deci.ai/super-gradients/docstring/training/models/#training.models.detection_models.pp_yolo_e.post_prediction_callback.PPYoloEPostPredictionCallback) |


## Training

The easiest way to start training any mode in SuperGradients is to use a pre-defined recipe. In this tutorial, we will see how to train `YOLOX-S` model, other models can be trained by analogy.

### Prerequisites

1. You have to install SuperGradients first. Please refer to the [Installation](installation.md) section for more details.
2. Prepare the COCO dataset as described in the [Computer Vision Datasets Setup](https://docs.deci.ai/super-gradients/src/super_gradients/training/datasets/Dataset_Setup_Instructions/) under Detection Datasets section. 

After you meet the prerequisites, you can start training the model by running from the root of the repository:

### Training from recipe

```bash
python -m super_gradients.train_from_recipe --config-name=coco2017_yolox multi_gpu=Off num_gpus=1
```

Note, the default configuration for this recipe is to use 8 GPUs in DDP mode. This hardware configuration may not be for everyone, so in the example above we override GPU settings to use a single GPU.
It is highly recommended to read through the recipe file [coco2017_yolox](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/coco2017_yolox.yaml) to get better understanding of the hyperparameters we use here.
If you're unfamiliar with config files, we recommend you to read the [Configuration Files](configuration_files.md) part first.

### Datasets

There are several well-known datasets for object detection: COCO, Pascal, etc. 
SuperGradients provides ready-to-use dataloaders for the COCO dataset [COCODetectionDataset](https://docs.deci.ai/super-gradients/docstring/training/datasets/#training.datasets.detection_datasets.coco_detection.COCODetectionDataset) 
and more general `DetectionDataset` implementation that you can subclass from for your specific dataset format.

If you want to load the dataset outside of a yaml training, do:
```python
from super_gradients.training import dataloaders


data_dir = "/path/to/coco_dataset_dir"
train_dataloader = dataloaders.get(name='coco2017_train',
                                   dataset_params={"data_dir": data_dir},
                                   dataloader_params={'num_workers': 2}
                                   )
val_dataloader = dataloaders.get(name='coco2017_val',
                                 dataset_params={"data_dir": data_dir},
                                 dataloader_params={'num_workers': 2}
                                 )
```

### Metrics

A typical metric for object detection is mean average precision, mAP for short. 
It is calculated for a specific IoU level which defines how tightly a predicted box must intersect with a ground truth box to be considered a true positive.
Both one value and a range can be used as IoU, where a range refers to an average of mAPs for each IoU level.
The most popular metric for mAP on COCO is mAP@0.5:0.95, SuperGradients provide its implementation [DetectionMetrics](https://docs.deci.ai/super-gradients/docstring/training/metrics/#training.metrics.detection_metrics.DetectionMetrics).
It is written to be as close as possible to the  official metric implementation from [COCO API](https://pypi.org/project/pycocotools/), while being much faster and DDP-friendly.

In order to use `DetectionMetrics` you have to pass a so-called `post_prediction_callback` to the metric, which is responsible for the postprocessing of the model's raw output into final predictions and is explained below. 

### Postprocessing

Postprocessing refers to a process of transforming the model's raw output into final predictions. Postprocessing is also model-specific and depends on the model's output format.
For `YOLOX` model, the postprocessing step is implemented in [YoloPostPredictionCallback](https://docs.deci.ai/super-gradients/docstring/training/models/#training.models.detection_models.yolo_base.YoloPostPredictionCallback) class.
It can be passed into a `DetectionMetrics` as a `post_prediction_callback`. 
The postprocessing of all detection models involves non-maximum suppression (NMS) which filters dense model's predictions and leaves only boxes with the highest confidence and suppresses boxes with very high overlap 
based on the assumption that they likely belong to the same object. Thus, a confidence threshold and an IoU threshold must be passed into the postprocessing object.

```python
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback


post_prediction_callback = YoloPostPredictionCallback(conf=0.001, iou=0.6)
```

### Visualization

Visualization of the model predictions is a very important part of the training process for any computer vision task. 
By visualizing the predicted boxes, developers and researchers can identify errors or inaccuracies in the model's output and adjust the model's architecture or training data accordingly.

SuperGradients provide an implementation of [DetectionVisualizationCallback](https://docs.deci.ai/super-gradients/docstring/training/utils/#training.utils.callbacks.callbacks.DetectionVisualizationCallback). 
You can use this callback in your training pipeline to visualize predictions during training. For this, just add it to `training_hyperparams.phase_callbacks` in your yaml.
During training, the callback will generate a visualization of the model predictions and save it to the TensorBoard or Weights & Biases depending on which logger you
are using (Default is Tensorboard). 

If you would like to do the visualization outside of training you can use `DetectionVisualization` class as follows:
```python
import torch
import numpy as np

from super_gradients.training import models
from super_gradients.training.utils.detection_utils import DetectionVisualization
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST


def my_undo_image_preprocessing(im_tensor: torch.Tensor) -> np.ndarray:
    im_np = im_tensor.cpu().numpy()
    im_np = im_np[:, ::-1, :, :].transpose(0, 2, 3, 1)
    im_np *= 255.0
    return np.ascontiguousarray(im_np, dtype=np.uint8)


model = models.get("yolox_s", pretrained_weights="coco", num_classes=80)
imgs, targets = next(iter(train_dataloader))
preds = YoloPostPredictionCallback(conf=0.1, iou=0.6)(model(imgs))
DetectionVisualization.visualize_batch(imgs, preds, targets, batch_name='train', class_names=COCO_DETECTION_CLASSES_LIST,
                                       checkpoint_dir='/path/for/saved_images/', gt_alpha=0.5,
                                       undo_preprocessing_func=my_undo_image_preprocessing)
```
The function you pass as `undo_preprocessing_func` will define how to undo dataset transforms and return the image back into its initial formal (BGR, uint8).
This also allows you to test the correctness of your dataset implementation, since it saves images after they go through transforms. This may be especially useful for a train set with heavy augmentation transforms. You can see both the predictions and the ground truth, and give the ground truth box the desired opacity. 

The saved train image for a dataset with a mosaic transform should look something like this:

![train_24](images/train_24.jpg)

### Let's train!

As stated above, training can be launched with just one command. For the curious ones, let's see how all the components we've just discussed fall into place in one yaml.
```yaml
# coco2017_yolox
defaults:
  - training_hyperparams: coco2017_yolox_train_params
  - dataset_params: coco_detection_dataset_params
  - arch_params: yolox_s_arch_params
  - checkpoint_params: default_checkpoint_params
  - _self_
```
These are the actual components of [coco2017_yolox.yaml](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/coco2017_yolox.yaml)
The dataset parameters are defined in `dataset_params:` and are eventually passed into coco2017_train/val dataset mentioned above in the [Datasets](ObjectDetection.md#datasets) section 

The metric is part of `training_hyperparams` and so it's stated in the [coco2017_yolox_train_params.yaml](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/training_hyperparams/coco2017_yolox_train_params.yaml) with:

```yaml
valid_metrics_list:
  - DetectionMetrics:
      normalize_targets: True
      post_prediction_callback:
        _target_: super_gradients.training.models.detection_models.yolo_base.YoloPostPredictionCallback
        iou: 0.65
        conf: 0.01
      num_cls: 80
```

Notice how `YoloPostPredictionCallback` is passed as a `post_prediction_callback`.

A visualization belongs to `training_hyperparams` as well, specifically to the `phase_callbacks` list, as follows:
```yaml
phase_callbacks:
  - DetectionVisualizationCallback:
      phase:
        _target_: super_gradients.training.utils.callbacks.callbacks.Phase
        value: VALIDATION_EPOCH_END
      freq: 1
      post_prediction_callback:
        _target_: super_gradients.training.models.detection_models.yolo_base.YoloPostPredictionCallback
        iou: 0.65
        conf: 0.01
      classes: [
          "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush"
      ]
```
By default, this callback is not part of the yaml, so you can add it yourself if you prefer.

Using the provided yaml, SuperGradients can instantiate all the components and launch training from config:
```python
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
    name=get_param(cfg, "train_dataloader"),
    dataset_params=cfg.dataset_params.train_dataset_params,
    dataloader_params=cfg.dataset_params.train_dataloader_params,
)

val_dataloader = dataloaders.get(
    name=get_param(cfg, "val_dataloader"),
    dataset_params=cfg.dataset_params.val_dataset_params,
    dataloader_params=cfg.dataset_params.val_dataloader_params,
)

recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}
# TRAIN
res = trainer.train(
    model=model,
    train_loader=train_dataloader,
    valid_loader=val_dataloader,
    training_params=cfg.training_hyperparams,
    additional_configs_to_log=recipe_logged_cfg,
)
```
It is convenient to trigger it with [train_from_recipe.py](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py), but you can do the same in your project by constructing the desired objects directly.


## How to connect your own dataset

To add a new dataset to SuperGradients, you need to implement a few things:

- Implement a new dataset class
- Add a configuration file

Let's unwrap each of the steps

### Implement a new dataset class

To train an existing architecture on a new dataset one needs to implement the dataset class first.
It is generally a good idea to subclass from `DetectionDataset` as it comes with a few useful features, 
such as subclassing, caching, extra sample loading necessary for complex transform like mosaic or mixup, etc. It requires you to implement only a few methods for files loading. 
If you prefer, you can use `torch.utils.data.Dataset` as well.

A minimal implementation of a `DetectionDataset` subclass class should look similar to this:

```python
import os.path
from typing import Tuple, List, Dict, Union, Any, Optional

import numpy as np
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.transforms.transforms import DetectionTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat


MY_CLASSES = ['cat', 'dog', 'donut']


@register_dataset("MyNewDetectionDataset")
class MyNewDetectionDataset(DetectionDataset):
    def __init__(self, data_dir: str, samples_dir: str, targets_dir: str, input_dim: Tuple[int, int],
                 transforms: List[DetectionTransform], max_num_samples: int = None, 
                 class_inclusion_list: Optional[List[str]] = None, **kwargs):
        self.sample_paths = None
        self.samples_sub_directory = samples_dir
        self.targets_sub_directory = targets_dir

        # setting cache as False to be able to load non-resized images and crop in one of transforms
        super().__init__(data_dir=data_dir, input_dim=input_dim,
                         original_target_format=DetectionTargetsFormat.LABEL_CXCYWH,
                         max_num_samples=max_num_samples,
                         class_inclusion_list=class_inclusion_list, transforms=transforms,
                         all_classes_list=MY_CLASSES, **kwargs)

    def _setup_data_source(self) -> int:
        """
        Set up the data source and store relevant objects as attributes.
        :return: Number of available samples (i.e. how many images we have)
        """

        samples_dir = os.path.join(self.data_dir, self.samples_sub_directory)
        labels_dir = os.path.join(self.data_dir, self.targets_sub_directory)
        sample_names = [n for n in sorted(os.listdir(samples_dir)) if n.endswith(".jpg")]
        label_names = [n for n in sorted(os.listdir(labels_dir)) if n.endswith(".txt")]
        assert len(sample_names) == len(label_names), f"Number of samples: {len(sample_names)}, " \
                                                      f"doesn't match the number of labels {len(label_names)}"
        self.samples_targets_tuples_list = []
        for sample_name in sample_names:
            sample_path = os.path.join(samples_dir, sample_name)
            label_path = os.path.join(labels_dir, sample_name.replace(".jpg", ".txt"))

            if os.path.exists(sample_path) and os.path.exists(label_path):
                self.samples_targets_tuples_list.append((sample_path, label_path))
            else:
                raise AssertionError(f"Sample and/or target file(s) not found "
                                     f"(sample path: {sample_path}, target path: {label_path})")

        return len(self.samples_targets_tuples_list)

    def _load_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Load annotations associated to a specific sample.
        Please note that the targets should be resized according to self.input_dim!
        :param sample_id:   Id of the sample to load annotations from.
        :return:            Annotation, a dict with any field. Has to include "image", "target", and "resized_img_shape"
        """
        sample_path, target_path = self.samples_targets_tuples_list[sample_id]

        with open(target_path, 'r') as targets_file:
            lines = targets_file.read().splitlines()
            target = np.array([x.strip().strip(',').split(',') for x in lines], dtype=np.float32)

        res_target = np.array(target) if len(target) != 0 else np.zeros((0, 5))  # cls, cx, cy, w, h
        annotation = {
            'img_path': os.path.join(self.data_dir, sample_path),
            'target': res_target,
            'resized_img_shape': None
        }

        return annotation
```
Note the addition of `@register_dataset` decorator. This makes SuperGradients recognize your dataset so that you can use its name directly in a yaml.
Since detection labels often contain different number of boxes per image, targets are padded with 0s, which allows to use them in a batch. 
They are later removed by a [DetectionCollateFN](https://docs.deci.ai/super-gradients/docstring/training/utils/#training.utils.detection_utils.DetectionCollateFN) which prepends all targets with an index in a batch and stacks them together.


### Add a configuration file

Create new `my_new_dataset_params.yaml` file under `dataset_params` folder and add your dataset and dataloader parameters:

```yaml
# my_new_dataset_params.yaml
root_dir: /path/to/my_data/

train_dataset_params:
  data_dir: ${dataset_params.root_dir}
  samples_dir: my-data-train/images
  targets_dir: my-data-train/annotations
  input_dim: [640, 640]
  transforms:
    - DetectionHSV:
        prob: 1.0                       # probability to apply HSV transform
        hgain: 5                        # HSV transform hue gain (randomly sampled from [-hgain, hgain])
        sgain: 30                       # HSV transform saturation gain (randomly sampled from [-sgain, sgain])
        vgain: 30                       # HSV transform value gain (randomly sampled from [-vgain, vgain])
        bgr_channels: [ 0, 1, 2 ]
    - DetectionStandardize:
        max_value: 255.
    - DetectionTargetsFormatTransform:
        input_format: XYWH_LABEL
        output_format: LABEL_CXCYWH

train_dataloader_params:
  dataset: MyNewDetectionDataset
  batch_size: 64
  num_workers: 8
  drop_last: True
  shuffle: True
  pin_memory: True
  worker_init_fn:
    _target_: super_gradients.training.utils.utils.load_func
    dotpath: super_gradients.training.datasets.datasets_utils.worker_init_reset_seed
  collate_fn:
    _target_: super_gradients.training.utils.detection_utils.DetectionCollateFN

val_dataset_params:
  data_dir: ${dataset_params.root_dir}
  samples_dir: my-data-val/images
  targets_dir: my-data-val/annotations
  input_dim: [640, 640]
  transforms:
    - DetectionStandardize:
        max_value: 255.
    - DetectionTargetsFormatTransform:
        input_format: XYWH_LABEL
        output_format: LABEL_CXCYWH

val_dataloader_params:
  dataset: MyNewDetectionDataset
  batch_size: 64
  num_workers: 8
  drop_last: True
  pin_memory: True
  collate_fn:
    _target_: super_gradients.training.utils.detection_utils.DetectionCollateFN
```

In your training recipe add/change the following lines to:

```yaml
# my_train_recipe.yaml
defaults:
  - training_hyperparams: ...
  - dataset_params: my_new_dataset_params
  - arch_params: ...
  - checkpoint_params: ...
  - _self_
 
train_dataloader:
val_dataloader:
num_classes: 3
...
```

And you should be good to go!

## How to add a new model

To implement a new model, you need to add the following parts:

- Model architecture itself
- Postprocessing Callback

For a custom model, a good starting point would be a 
[CustomizableDetector](https://docs.deci.ai/super-gradients/docstring/training/models/#training.models.detection_models.customizable_detector.CustomizableDetector) 
class since it allows to configure a backbone, a neck and a head separately. See an example yaml of 
a model that uses it: [ssd_lite_mobilenetv2_arch_params](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/arch_params/ssd_lite_mobilenetv2_arch_params.yaml)

It is strongly advised to use the existing callbacks and to define your model's head such that it returns the same outputs. 
