# Image Segmentation

SuperGradients allows users to train models for semantic segmentation tasks.
The library includes pre-trained models, such as the Cityscapes PPLiteSeg model, and provides a simple interface for 
loading custom datasets. 

## Model zoo

SuperGradients includes a variety of pre-trained models for semantic segmentation tasks.

| Model Name     | Dataset    | IoU   | Training Recipe                                                                                                                                                                   | Resolution    |
|----------------|------------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| DDRNet 23      | Cityscapes | 80.26 | [cityscapes_ddrnet.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_ddrnet.yaml)             | [1024, 2048]  |
| DDRNet 23 Slim | Cityscapes | 78.01 | [cityscapes_ddrnet.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_ddrnet.yaml)             | [1024, 2048]  |
| DDRNet 39      | Cityscapes | 81.32 | [cityscapes_ddrnet.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_ddrnet.yaml)             | [1024, 2048]  |
| STDC1 Seg 50   | Cityscapes | 75.11 | [cityscapes_stdc_seg50.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_stdc_seg50.yaml)     | [512, 1024]   |
| STDC1 Seg 75   | Cityscapes | 76.87 | [cityscapes_stdc_seg75.yaml](https://github.com/Deci-AI/super-gradients/blob/6e89982649e62e9877a802cd1240464cd3b3b87b/src/super_gradients/recipes/cityscapes_stdc_seg75.yaml)     | [768, 1536]   |
| STDC2 Seg 50   | Cityscapes | 76.44 | [cityscapes_stdc_seg50.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_stdc_seg50.yaml)     | [512, 1024]   |
| STDC2 Seg 75   | Cityscapes | 78.93 | [cityscapes_stdc_seg75.yaml](https://github.com/Deci-AI/super-gradients/blob/6e89982649e62e9877a802cd1240464cd3b3b87b/src/super_gradients/recipes/cityscapes_stdc_seg75.yaml)     | [768, 1536]   |
| RegSeg 48      | Cityscapes | 78.15 | [cityscapes_regseg48.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_regseg48.yaml)         | [1024, 2048]  |
| PP-Lite T 50   | Cityscapes | 74.92 | [cityscapes_pplite_seg50.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_pplite_seg50.yaml) | [512, 1024]   |
| PP-Lite T 75   | Cityscapes | 77.56 | [cityscapes_pplite_seg75.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_pplite_seg75.yaml) | [512, 1024]   |
| PP-Lite B 50   | Cityscapes | 76.48 | [cityscapes_pplite_seg50.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_pplite_seg50.yaml) | [512, 1024]   |
| PP-Lite B 75   | Cityscapes | 78.52 | [cityscapes_pplite_seg75.yaml](https://github.com/Deci-AI/super-gradients/blob/95018d602ef7f65b37d5ec62e26a8ebbc5b2a7c8/src/super_gradients/recipes/cityscapes_pplite_seg75.yaml) | [512, 1024]   |

Latency and additional details of these models can be found in the [SuperGradients Model Zoo](https://docs.deci.ai/super-gradients/documentation/source/model_zoo.html).

## Loss functions

SuperGradients provides a variety of loss functions for training semantic segmentation tasks. 
All loss functions are implemented in PyTorch and can be found in the `super_gradients.training.losses` module. 
The following table summarizes the loss functions currently supported by SuperGradients.

| Loss function class                                                                                                                                                                     | Loss name in YAML | Description                                                          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|----------------------------------------------------------------------|
| [BCEDiceLoss](https://docs.deci.ai/super-gradients/docstring/training/losses.html#training.losses.bce_dice_loss.BCEDiceLoss)                                                            | bce_dice_loss     | Weighted average of BCE and Dice loss                                |
| [CrossEntropyLoss](https://docs.deci.ai/super-gradients/docstring/training/losses.html#training.losses.label_smoothing_cross_entropy_loss.CrossEntropyLoss) | cross_entropy     | Cross entropy loss with label smoothing support                      |
| [DiceLoss](https://docs.deci.ai/super-gradients/docstring/training/losses.html#training.losses.dice_loss.DiceLoss)                                                                      | N/A               | Dice loss for multiclass segmentation                                |
| [BinaryDiceLoss](https://docs.deci.ai/super-gradients/docstring/training/losses.html#training.losses.dice_loss.BinaryDiceLoss)                                                          | N/A               | Dice loss for binary segmentation                                    |
| [GeneralizedDiceLoss](https://docs.deci.ai/super-gradients/docstring/training/losses.html#training.losses.dice_loss.GeneralizedDiceLoss)                                                | N/A               | Generalized dice loss                                                |
| [DiceCEEdgeLoss](https://docs.deci.ai/super-gradients/docstring/training/losses.html#training.losses.dice_ce_edge_loss.DiceCEEdgeLoss)                                                  | dice_ce_edge_loss | Dice loss + Cross entropy loss + Edge loss                           |
| [SegKDLoss](https://docs.deci.ai/super-gradients/docstring/training/losses.html#training.losses.seg_kd_loss.SegKDLoss)                                                                  | N/A               | A loss function for knowledge distillation for semantic segmentation |

## Metrics

| Metric Class                                                                                                                              | Metric name in YAML | Description                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------|---------------------|------------------------------------------------------------------------|
| [PixelAccuracy](https://docs.deci.ai/super-gradients/docstring/training/metrics.html#training.metrics.segmentation_metrics.PixelAccuracy) | PixelAccuracy       | The ratio of correctly classified pixels to the total number of pixels |
| [IoU](https://docs.deci.ai/super-gradients/docstring/training/metrics.html#training.metrics.segmentation_metrics.IoU)                     | IoU                 | Calculate the Jaccard index for multilabel tasks.                      |
| [Dice](https://docs.deci.ai/super-gradients/docstring/training/metrics.html#training.metrics.segmentation_metrics.Dice)                   | Dice                | Calculate the Dice index for multilabel tasks.                         |                                                |
| Binary IoU                                                                                                                                | BinaryIOU           | Calculate the Jaccard index for binary segmentation task.              |
| BinaryDice                                                                                                                                | BinaryDice          | Calculate the Dice index for binary segmentation task.                 |

See [Metrics](https://docs.deci.ai/super-gradients/documentation/source/Metrics.html) page for additional details of using metrics in SuperGradients.

## Datasets

SuperGradients provides a number of ready to use datasets for semantic segmentation tasks and corresponding data loaders.

| Dataset          | Dataset Class                                                                                                                                                                                              | train dataloader              | val dataloader              |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|-----------------------------|
| COCO             | [CoCoSegmentationDataSet](https://docs.deci.ai/super-gradients/docstring/training/datasets.html#training.datasets.segmentation_datasets.coco_segmentation.CoCoSegmentationDataSet)                         | coco_segmentation_train       | coco_segmentation_val       |
| Cityscapes       | [CityscapesDataset](https://docs.deci.ai/super-gradients/docstring/training/datasets.html#training.datasets.segmentation_datasets.cityscape_segmentation.CityscapesDataset)                                | cityscapes_train              | cityscapes_val              |
| Pascal VOC       | [PascalVOC2012SegmentationDataSet](https://docs.deci.ai/super-gradients/docstring/training/datasets.html#training.datasets.segmentation_datasets.pascal_voc_segmentation.PascalVOC2012SegmentationDataSet) | pascal_voc_segmentation_train | pascal_voc_segmentation_val |
| Supervisely      | [SuperviselyPersonDataset](https://docs.deci.ai/super-gradients/docstring/training/datasets.html#training.datasets.segmentation_datasets.supervisely_persons_segmentation.SuperviselyPersonsDataset)       | supervisely_persons_train     | supervisely_persons_val     |
| Mapillary Vistas | [MapillaryDataset](https://docs.deci.ai/super-gradients/docstring/training/datasets.html#training.datasets.segmentation_datasets.mapillary_dataset.MapillaryDataset)                                       | mapillary_train               | mapillary_val               |

In the next section we will demonstrate how to use these datasets and dataloaders to train a segmentation model using SuperGradients.

## How to train a segmentation model using Super Gradients

In the tutorial provided, we demonstrate how to fine-tune PPLiteSeg on a subset of the Supervisely dataset.
You can run the following code in our [google collab](https://colab.research.google.com/drive/1d7cU0NsUj7jnOF1YSap_DH9r79G3-Cr4?usp=sharing#scrollTo=GqH4VGMroWec).

## Load a dataset
In this example we will work with supervisely-persons. If it's the first time you are using this dataset, or if you want to use another dataset please check out [dataset setup instructions](Data.md)
```py
from super_gradients.training import dataloaders

root_dir = '/path/to/supervisely_dataset_dir'

train_loader = dataloaders.supervisely_persons_train(dataset_params={"root_dir": root_dir}, dataloader_params={})
valid_loader = dataloaders.supervisely_persons_val(dataset_params={"root_dir": root_dir}, dataloader_params={})
```


### Visualization
Let's visualize what we've got there.

We have images and labels, with the default batch size of 256 for training.
```py
from PIL import Image
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms import ToTensor, ToPILImage, Resize
import numpy as np
import torch

def plot_seg_data(img_path: str, target_path: str):
  image = (ToTensor()(Image.open(img_path).convert('RGB')) * 255).type(torch.uint8)
  target = torch.from_numpy(np.array(Image.open(target_path))).bool()
  image = draw_segmentation_masks(image, target, colors="red", alpha=0.4)
  image = Resize(size=200)(image)
  display(ToPILImage()(image))

for i in range(4, 7):
  img_path, target_path = train_loader.dataset.samples_targets_tuples_list[i]
  plot_seg_data(img_path, target_path)
```
![segmentation_target.png](segmentation_target.png)


## Load the model from modelzoo

Create a PPLiteSeg nn.Module, with 1 class segmentation head classifier. For simplicity `use_aux_head` is set as `False`
and extra Auxiliary heads aren't used for training.
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
Notes

- SG includes implementations of 
[many different architectures](https://github.com/Deci-AI/super-gradients#implemented-model-architectures).
- Most of these architectures have [pretrained checkpoints](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/Computer_Vision_Models_Pretrained_Checkpoints.md) so feel free to experiment!
- You can use any torch.nn.module model with SuperGradients!


### Setup training parameters

The training parameters includes loss, metrics, learning rates and much more. You can check out the [default training parameters](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml).
For this task, we will train for 30 epoch, using Binary IoU using the SGD optimizer.

```py
from super_gradients.training.metrics.segmentation_metrics import BinaryIOU

train_params = {
    "max_epochs": 30,
    "lr_mode": "CosineLRScheduler",
    "initial_lr": 0.005,
    "lr_warmup_epochs": 5,
    "multiply_head_lr": 10,
    "optimizer": "SGD",
    "loss": "BCEDiceLoss",
    "ema": True,
    "zero_weight_decay_on_bias_and_bn": True,
    "average_best_models": True,
    "metric_to_watch": "target_IOU",
    "greater_metric_to_watch_is_better": True,
    "train_metrics_list": [BinaryIOU()],
    "valid_metrics_list": [BinaryIOU()],
    "loss_logging_items_names": ["loss"],
}
```


### Launch Training
The Trainer in SuperGradient takes care of the entire training and validation process. 

It serves as a convenient and efficient tool to handle all the details of the training process, allowing you to focus on the development of your model.

```py
from super_gradients import Trainer

trainer = Trainer(
    experiment_name="segmentation_example",     # Your experiment checkpoints and logs will be saved in a folder names after the experiment_name.
    ckpt_root_dir='/path/to/experiment/folder'  # Path to the folder where you want to save all of your experiments.
)

trainer.train(model=model, training_params=training_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```


## Visualize the results

```py
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ToPILImage

pre_proccess = Compose([
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
])

demo_img_path = "/home/data/supervisely-persons/images/ache-adult-depression-expression-41253.png"
img = Image.open(demo_img_path)
# Resize the image and display
img = Resize(size=(480, 320))(img)
display(img)

# Run pre-proccess - transforms to tensor and apply normalizations.
img_inp = pre_proccess(img).unsqueeze(0).cuda()

# Run inference
mask = model(img_inp)

# Run post-proccess - apply sigmoid to output probabilities, then apply hard
# threshold of 0.5 for binary mask prediction. 
mask = torch.sigmoid(mask).gt(0.5).squeeze()
mask = ToPILImage()(mask.float())
display(mask)
```
![segmentation_prediction.png](segmentation_prediction.png)


## Going further

### Troubleshooting

If you encounter any issues, please check out our [troubleshooting guide](https://docs.deci.ai/super-gradients/documentation/source/troubleshooting.html).

### How to launch on multiple GPUs (DDP) ?

Please check out our tutorial on [how to use multiple GPUs'](https://docs.deci.ai/super-gradients/documentation/source/device.html#4-ddp-distributed-data-parallel)

### How to train models with limited GPU memory?

In case you have a GPU with limited memory, you can use the gradients accumulation technique to "fake" larger batch sizes.
This is not 100% equivalent to training with larger batch sizes, but it is a good approximation. 
You can set the desired number of batches to accumulate by changing the `training_hyperparams.batch_accumulate` parameter.
