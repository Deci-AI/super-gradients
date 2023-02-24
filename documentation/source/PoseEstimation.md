# Pose Estimation

Pose estimation is a computer vision task that involves estimating the position and orientation of objects or people in images or videos. 
It typically involves identifying specific keypoints or body parts, such as joints, and determining their relative positions and orientations. 
Pose estimation has numerous applications, including robotics, augmented reality, human-computer interaction, and sports analytics.

Top-down and bottom-up are two commonly used approaches in pose estimation. The main difference between top-down and bottom-up pose estimation approaches is the order in which the pose is estimated.

In a **top-down approach**, an object detection model is used to identify the object of interest, such as a person or a car, and a separate pose estimation model is used to estimate the pose of the object.

In contrast, a **bottom-up** approach first identifies individual body parts or joints and then connects them to form a complete pose.

In summary, top-down approach starts with detecting an object and then estimates its pose, while bottom-up approach first identifies the body parts and then forms a complete pose.

SuperGradients provides a recipe for training a SOTA **bottom-up** model called `DEKR` from ["Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression"](https://arxiv.org/abs/2104.02300) paper.

## Datasets

There are several well-known datasets that contain pose estimation annotations for different tasks: COCO, MPII Human Pose, Hands in the Wild, CrowdPose, etc.

SuperGradients provide ready-to-use dataloaders for the COCO dataset `COCOKeypointsDataset` and more general `KeypointsDataset` implementation that you can subclass for your specific dataset format.

### Target generators

The target generators are responsible for generating the target tensors for the model. 
Implementation of target generator is model-specific. 
So each model may require its own target generator implementation that is compatible with model's output. 
The goal of this class is to transform ground-truth annotations into the format that is suitable for computing a loss and training a model.

SuperGradients provide implementation of `DEKRTargetGenerator` that is compatible with `DEKR` model.

If you need to implement your own target generator, please refer to documentation of `KeypointsTargetsGenerator` base class.

```py
from super_gradients.training import dataloaders
from super_gradients.training.datasets.pose_estimation_datasets.target_generators import DEKRTargetsGenerator
root_dir = '/path/to/coco2017'

target_generator = DEKRTargetsGenerator(
    ...
)
train_loader = dataloaders.coco2017_pose_train(dataset_params={"root_dir": root_dir, "target_generator": target_generator}, dataloader_params={})
valid_loader = dataloaders.coco2017_pose_val(dataset_params={"root_dir": root_dir, "target_generator": target_generator}, dataloader_params={})
```

## Load the model from modelzoo

Create a DEKR-W32 model, with 1 class segmentation head classifier. 

```py
from super_gradients.training import models
from super_gradients.common.object_names import Models

# The model is a torch.nn.module 
model = models.get(
    model_name=Models.DEKR_CUSTOM,      # You can use any model listed in the Models.<Name>
    pretrained_weights="coco"           # Drop this line to train from scratch
)
```


## Loss function

Implemented loss functions in SuperGradients for training pose estimation models:
 
* `DEKRLoss` - loss function for `DEKR` model.


## Metrics

A typical metric for pose estimation is the average precision (AP) and average recall (AR). 
SuperGradients provide implementation of `PoseEstimationMetrics` to compute AP/AR scores.

### Postprocessing

Postprocessing is a process of transforming model's raw output into a final predictions. Postprocessing is model-specific and depends on the model's output format. 
For `DEKR` model, postprocessing step is implemented in `DEKRPoseEstimationDecodeCallback` class. When instantiating the metric, one have to pass postprocessing callback as an argument:

```yaml
training_hyperparams:
    valid_metrics_list:
      - PoseEstimationMetrics:
          num_joints: ${dataset_params.num_joints}
          oks_sigmas: ${dataset_params.oks_sigmas}
          max_objects_per_image: 20
          post_prediction_callback:
            _target_: super_gradients.training.utils.pose_estimation.DEKRPoseEstimationDecodeCallback
            max_num_people: 20
            keypoint_threshold: 0.05
            nms_threshold: 0.05
            nms_num_threshold: 8
            output_stride: 4
            apply_sigmoid: False
```

## Visualization

Visualization of the model predictions is a very important part of the training process for pose estimation models. 
By visualizing the predicted poses, developers and researchers can identify errors or inaccuracies in the model's output and adjust the model's architecture or training data accordingly.

Overall, visualization is an important tool for improving the accuracy and usability of pose estimation models, both during development and in real-world applications.

SuperGradients provide implementation of `DEKRVisualizationCallback` to visualize predictions for `DEKR` model. 
You can use this callback in your training pipeline to visualize predictions during training. To enable this callback, add the following lines to your training YAML recipe:

```yaml
training_hyperparams:
  resume: ${resume}
  phase_callbacks:
    - DEKRVisualizationCallback:
        phase:
          _target_: super_gradients.training.utils.callbacks.callbacks.Phase
          value: TRAIN_BATCH_END
        prefix: "train_"
        mean: [ 0.485, 0.456, 0.406 ]
        std: [ 0.229, 0.224, 0.225 ]
        apply_sigmoid: False

    - DEKRVisualizationCallback:
        phase:
          _target_: super_gradients.training.utils.callbacks.callbacks.Phase
          value: VALIDATION_BATCH_END
        prefix: "val_"
        mean: [ 0.485, 0.456, 0.406 ]
        std: [ 0.229, 0.224, 0.225 ]
        apply_sigmoid: False
```

## Implementing your own model

To implement a new model, you may need to implement the following classes:

* Dataset class
* Target Generator
* Postprocessing Callback
* (Optional) Visualization Callback

A custom dataset class should inherit from `KeypointsDataset` base class which provides a common interface, transforms, and other useful methods.

A custom target generator class should inherit from `KeypointsTargetsGenerator` base class which provides a protocol for generating target tensors for the ground-truth keypoints.

A custom postprocessing callback class should inherit from `PoseEstimationDecodeCallback` base class which provides a protocol for transforming model's raw output into a final predictions.

A custom visualization callback class can inherit from `PhaseCallback` or `Callback` base class to generate visualization of the model predictions.
