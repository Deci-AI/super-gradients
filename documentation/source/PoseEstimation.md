# Pose Estimation

Pose estimation is a computer vision task that involves estimating the position and orientation of objects or people in images or videos. 
It typically involves identifying specific keypoints or body parts, such as joints, and determining their relative positions and orientations. 
Pose estimation has numerous applications, including robotics, augmented reality, human-computer interaction, and sports analytics.

Top-down and bottom-up are two commonly used approaches in pose estimation. The main difference between top-down and bottom-up pose estimation approaches is the order in which the pose is estimated.

In a **top-down approach**, an object detection model is used to identify the object of interest, such as a person or a car, and a separate pose estimation model is used to estimate the pose of the object.

In contrast, a **bottom-up** approach first identifies individual body parts or joints and then connects them to form a complete pose.

In summary, top-down approach starts with detecting an object and then estimates its pose, while bottom-up approach first identifies the body parts and then forms a complete pose.

SuperGradients provides a recipe for training a SOTA **bottom-up** model called DEKR.

## Datasets

There are several well-known datasets that contain pose estimation annotations for different tasks. Here are some of the most popular ones:

* COCO (Common Objects in Context) Keypoints: This is a large-scale dataset for human pose estimation. It contains over 200,000 images with more than 250,000 person instances labeled with keypoints.
* MPII Human Pose: This dataset contains around 25,000 images of people in various poses, including sitting, standing, and bending. It provides annotations for 16 different body joints and has been used for benchmarking various human pose estimation algorithms.
* Pascal VOC: This is a popular dataset for object detection, segmentation, and classification. It includes annotations for object bounding boxes, object categories, and object parts, which can be used for pose estimation.
* NYU Depth v2: This is a dataset for indoor scene understanding, including human pose estimation. It contains 1449 densely labeled pairs of aligned RGB and depth images, with annotations for body joints.
* Hands in the Wild: This dataset contains images of hands in various natural environments, with annotations for hand poses and hand-object interactions.

These datasets have been widely used for training and evaluating various pose estimation models and algorithms, and have contributed significantly to the advancement of pose estimation research.

In this tutorial we focus on using COCO dataset.

```py
from super_gradients.training import dataloaders

root_dir = '/path/to/coco2017'

train_loader = dataloaders.coco2017_pose_train(dataset_params={"root_dir": root_dir}, dataloader_params={})
valid_loader = dataloaders.coco2017_pose_val(dataset_params={"root_dir": root_dir}, dataloader_params={})
```

## Load the model from modelzoo

Create a DEKR-W32 model, with 1 class segmentation head classifier. 
For simplicity `use_aux_head` is set as `False` and extra Auxiliary heads aren't used for training.

```py
from super_gradients.training import models
from super_gradients.common.object_names import Models

# The model is a torch.nn.module 
model = models.get(
    model_name=Models.PP_LITE_T_SEG75,      # You can use any model listed in the Models.<Name>
    arch_params={"use_aux_heads": False},
    num_classes=1,                          # Change this if you work on another dataset with more classes
    pretrained_weights="cityscapes"         # Drop this line to train from scratch
)
```
