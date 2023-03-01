# Pose Estimation

Pose estimation is a computer vision task that involves estimating the position and orientation of objects or people in images or videos. 
It typically involves identifying specific keypoints or body parts, such as joints, and determining their relative positions and orientations. 
Pose estimation has numerous applications, including robotics, augmented reality, human-computer interaction, and sports analytics.

Top-down and bottom-up are two commonly used approaches in pose estimation. The main difference between top-down and bottom-up pose estimation approaches is the order in which the pose is estimated.

In a **top-down approach**, an object detection model is used to identify the object of interest, such as a person or a car, and a separate pose estimation model is used to estimate the keypoints of the object.

In contrast, a **bottom-up** approach first identifies individual body parts or joints and then connects them to form a complete pose.

In summary, top-down approach starts with detecting an object and then estimates its pose, while bottom-up approach first identifies the body parts and then forms a complete pose.

## Implemented models

| Model                                    | Model class                                                                                                                                                          | Target Generator                                                                                                                                                      | Loss Class                                                                                                     | Decoding Callback                                                                                                                                                                        | Visualization Callback                                                                                                                                                            |
|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
| [DEKR](https://arxiv.org/abs/2104.02300) | [DEKRPoseEstimationModel](https://docs.deci.ai/super-gradients/docstring/training/models/#training.models.pose_estimation_models.dekr_hrnet.DEKRPoseEstimationModel) | [DEKRTargetsGenerator](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/datasets/pose_estimation_datasets/target_generators.py#L8) | [DEKRLoss](https://docs.deci.ai/super-gradients/docstring/training/losses/#training.losses.dekr_loss.DEKRLoss) | [DEKRPoseEstimationDecodeCallback](https://docs.deci.ai/super-gradients/docstring/training/utils/#training.utils.pose_estimation.dekr_decode_callbacks.DEKRPoseEstimationDecodeCallback) | [DEKRVisualizationCallback](https://docs.deci.ai/super-gradients/docstring/training/utils/#training.utils.pose_estimation.dekr_visualization_callbacks.DEKRVisualizationCallback) |

## Training

For the sake of being specific in this tutorial, we will consider the training of `DEKR` model in further explanations.
The easiest way to start training a pose estimation model is to use a recipe from SuperGradients. 

```bash
# Note you may need to download ImageNet pretrained weights for HRNet backbone to obtain a on-par performance with the paper
python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=coco2017_pose_dekr_w32
```

If you're unfamiliar with config files, we recommend you to read the [Configuration Files](https://docs.deci.ai/super-gradients/documentation/source/configuration_files/) part first.

The start of the config file looks like this:

```yaml
defaults:
  - training_hyperparams: coco2017_dekr_pose_train_params
  - dataset_params: coco_pose_estimation_dekr_dataset_params
  - arch_params: dekr_w32_arch_params
  - checkpoint_params: default_checkpoint_params
  - _self_
```

Here we define the default values for the following parameters:
* `training_hyperparams` - These are our training hyperparameters. Things learning rate, optimizer, use of mixed precision, EMA and other training parameters are defined here. 
    You can refer to the [default_train_params.yaml](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml) for more details.
   In our example we use  [coco2017_dekr_pose_train_params.yaml](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/training_hyperparams/coco2017_dekr_pose_train_params.yaml) that sets
   training parameters as in [DEKR](https://arxiv.org/abs/2104.02300) paper.
* `dataset_params` - These are the parameters for the training on COCO2017. The dataset configuration sets the dataset transformations (augmentations & preprocessing) and [target generator](https://docs.deci.ai/super-gradients/docstring/training/datasets/#training.datasets.pose_estimation_datasets.target_generators.DEKRTargetsGenerator) for training the model.
* `arch_params` - These are the parameters for the model architecture. In our example we use [DEKRPoseEstimationModel](https://docs.deci.ai/super-gradients/docstring/training/models/#training.models.pose_estimation_models.dekr_hrnet.DEKRPoseEstimationModel) that is a HRNet-based model with DEKR decoder.
* `checkpoint_params` - These are the default parameters for resuming of training and using pretrained checkpoints. 
You can refer to the [default_checkpoint_params.yaml](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/checkpoint_params/default_checkpoint_params.yaml).
 
### Datasets

There are several well-known datasets for pose estimation: COCO, MPII Human Pose, Hands in the Wild, CrowdPose, etc. 
SuperGradients provide ready-to-use dataloaders for the COCO dataset [COCOKeypointsDataset](https://docs.deci.ai/super-gradients/docstring/training/datasets/#training.datasets.pose_estimation_datasets.coco_keypoints.COCOKeypointsDataset) 
and more general `KeypointsDataset` implementation that you can subclass from for your specific dataset format.

### Target generators

The target generators are responsible for generating the target tensors for the model. 
Implementation of the target generator is model-specific and usually includes at least a multi-channel heatmap mask per joint. 

Each model may require its own target generator implementation that is compatible with model's output. 

All target generators should implement `KeypointsTargetsGenerator` interface as shown below. 
The goal of this class is to transform ground-truth annotations into a format that is suitable for computing a loss and training a model:

```py
# super_gradients.training.datasets.pose_estimation_datasets.target_generators.KeypointsTargetsGenerator

import abc
import numpy as np
from torch import Tensor
from typing import Union, Tuple, Dict

class KeypointsTargetsGenerator:
    @abc.abstractmethod
    def __call__(self, image: Tensor, joints: np.ndarray, mask: np.ndarray) -> Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]]:
        """
        Encode input joints into target tensors

        :param image: [C,H,W] Input image tensor
        :param joints: [Num Instances, Num Joints, 3] Last channel represents (x, y, visibility)
        :param mask: [H,W] Mask representing valid image areas. For instance, in COCO dataset crowd targets
                           are not used during training and corresponding instances will be zero-masked.
                           Your implementation may use this mask when generating targets.
        :return: Encoded targets
        """
        raise NotImplementedError()
```

SuperGradients provide implementation of [DEKRTargetGenerator](https://docs.deci.ai/super-gradients/docstring/training/datasets/#training.datasets.pose_estimation_datasets.target_generators.DEKRTargetsGenerator) that is compatible with `DEKR` model.

If you need to implement your own target generator, please refer to documentation of `KeypointsTargetsGenerator` base class. 

### Metrics

A typical metric for pose estimation is the average precision (AP) and average recall (AR). 
SuperGradients provide implementation of `PoseEstimationMetrics` to compute AP/AR scores.

The metric is implemented as a callback that is called after each validation step. Implementation of the metric is made as close as possible to official metric implementation from [COCO API](https://pypi.org/project/pycocotools/).
However, our implementation does NOT include computation of AP/AR scores per area range. It also natively support evaluation in DDP mode. 

It is worth noting that usually reported AP/AR scores in papers are obtained using TTA (test-time augmentation) and additional postprocessing on top of the main model. 

A horizontal flip is a common TTA technique that is used to increase accuracy of the predictions at the cost of running forward pass twice. 
Second common technique is a multi-scale approach when one perform inference additionally on 0.5x and 1.5x input resolution and aggregate predictions. 

When training model using SuperGradients, we use neither of these techniques. If you want to measure AP/AR scores using TTA you may want to write your own evaluation loop for that.

In order to use `PoseEstimationMetrics` you have to pass a so-called `post_prediction_callback` to the metric, which is responsible for postprocessing of the model's raw output into final predictions. 

### Postprocessing

Postprocessing refers to a process of transforming the model's raw output into final predictions. Postprocessing is also model-specific and depends on the model's output format. 
For `DEKR` model, the postprocessing step is implemented in [DEKRPoseEstimationDecodeCallback]((https://docs.deci.ai/super-gradients/docstring/training/utils/#training.utils.pose_estimation.dekr_decode_callbacks.DEKRPoseEstimationDecodeCallback)) class. 
When instantiating the metric, one has to pass a postprocessing callback as an argument:

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

### Visualization

Visualization of the model predictions is a very important part of the training process for pose estimation models. 
By visualizing the predicted poses, developers and researchers can identify errors or inaccuracies in the model's output and adjust the model's architecture or training data accordingly.

Overall, visualization is an important tool for improving the accuracy and usability of pose estimation models, both during development and in real-world applications.


SuperGradients provide an implementation of `DEKRVisualizationCallback` to visualize predictions for `DEKR` model. 
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

During training, the callback will generate a visualization of the model predictions and save it to the TensorBoard or Weights & Biases depending on which logger you
are using (Default is Tensorboard). And result will look like this:

![](images/pose_estimation_visualization_callback.png)

On the left side of the image there is input image with ground-truth keypoints overlay and on the right side there are same channel-wise sum of target and predicted heatmaps.

## Implementing your own model

To implement a new model, you may need to implement the following classes:

* Dataset class
* Target Generator
* Postprocessing Callback
* (Optional) Visualization Callback

A custom dataset class should inherit from `KeypointsDataset` base class which provides a common interface, transforms, and other useful methods.

A custom target generator class should inherit from `KeypointsTargetsGenerator` base class which provides a protocol for generating target tensors for the ground-truth keypoints.

A custom postprocessing callback class should inherit from `PoseEstimationDecodeCallback` base class which provides a protocol for transforming the model's raw output into a final prediction.

A custom visualization callback class can inherit from `PhaseCallback` or `Callback` base class to generate a visualization of the model predictions.
