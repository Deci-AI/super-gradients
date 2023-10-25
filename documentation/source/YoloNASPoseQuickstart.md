# YOLO-NAS-POSE Quickstart
<div>
<img src="images/yolo_nas_pose_frontier.png" width="750">
</div>

Deci’s leveraged its proprietary Neural Architecture Search engine (AutoNAC) to generate YOLO-NAS-POSE - a new object 
detection architecture that delivers the world’s best accuracy-latency performance. 

The YOLO-NAS-POSE model incorporates quantization-aware RepVGG blocks to ensure compatibility with post-training 
quantization,  making it very flexible and usable for different hardware configurations.

In this tutorial, we will go over the basic functionality of the YOLO-NAS-POSE model. 


## Instantiate a YOLO-NAS-POSE Model

```python
from super_gradients.training import models
from super_gradients.common.object_names import Models

yolo_nas_pose = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights="coco_pose")
```

## Predict

```python
prediction = yolo_nas_pose.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg")
prediction.show()
```
<div>
<img src="images/yolo_nas_pose_beatles-abbeyroad.png" width="750">
</div>

## Export to ONNX

```python
yolo_nas_pose.export("yolo_nas_pose.onnx")
```

Please follow our [Pose Estimation Models Export](models_export_pose.md) tutorial for more details.
