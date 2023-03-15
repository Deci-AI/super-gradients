#SuperGradients Deci Yolo Tutorial

In this tutorial we present how can one take our DeciYolo to production from the beginning to end!

The tutorial is divided into 6 sections:
- Brief description of DeciYolo model
- Creating DeciYolo model
- Performing validation on COCO2017
- Inference on an individual image
- Fine-tuning DeciYolo on XXX dataset
- Inference on an individaul image from XXX dataset
- Performing QAT on XXX dataset
- Exporting our model to TRT Engine format

Pre-requisites:
- [Training with configuration files](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/configuration_files.md)
- [QAT](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/ptq_qat.md)

## What is DeciYolo?

Object detection is a crucial task in computer vision, enabling machines to interact with the world around them. YOLO models have become popular for their real-time performance, with the latest versions being YOLO-v6 and YOLO-v8.
Deci, a deep learning company, has developed AutoNAC, a Neural Architecture Search technology that can automatically construct deep learning models optimized for running efficiently on any desired AI accelerator.
Deci applied AutoNAC to discover novel models for object detection called DeciYolos, which deliver state-of-the-art performance and benefit from advanced training techniques available through Deci's SuperGradients open-source training library.

## Instantiating a DeciYolo Model
For the sake of this tutorial, we will use the small variant of DeciYolo - DeciYoloS.
As with any other architecture, we use models.get() to instantiate our pretrained DeciYolo model:
```python
from super_gradients.training import models
deci_yolo = models.get("deciyolo_s", pretrained_weights="coco").eval()
```
## Performing validation on COCO2017

Next, we want to be sure that the pre-trained weights are loaded properly, so we perform validation on COCO2017 val.
To do so, we need to use the same dataset configuration (i.e, preprocessing, collate function etc) that was used for validating the model during it's training time.
SuperGradients dataloaders module makes it easy to do so since any datalaoder configuration used in our training recipes can be instantiated with a one-liner.
```python
from super_gradients.training.dataloaders import coco2017_val_deci_yolo
from super_gradients import Trainer
from super_gradients.training import models

deci_yolo = models.get("deciyolo_s", pretrained_weights="coco").eval()
coco_val_dl = coco2017_val_deci_yolo()
trainer = Trainer(experiment_name="validation_deci_yolo")
#TODO ADD METRICS AND CALL TEST
```

## Inference on an individual image

To run inference on a single image, we have to make a few steps, which we cover in detail below. This should give you an in-depth understanding of the data pre-/post- processing steps required to get the detections.
Let's download our test image first:


```python
import requests
from PIL import Image

url = "https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg"
original_image = Image.open(requests.get(url, stream=True).raw)
original_image
```

DeciYolo expects the input image to have a size of 640x640. This ensures optimal model accuracy since it matches the size of the images on which the model was trained. As a rule of thumb - you always want to use the same resolution for training and inference.
For the sake of simplicity, we resize the input image without preserving an aspect ratio. This should be fine since during training aspect ratio was also not preserved.
