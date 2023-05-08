# Image Segmentation

SuperGradients allows users to train models for semantic segmentation tasks.
The library includes pre-trained models, such as the Cityscapes PPLiteSeg model, and provides a simple interface for 
loading custom datasets. 

In the tutorial provided, we demonstrate how to fine-tune PPLiteSeg on a subset of the Supervisely dataset.
You can run the following code in our [google collab](https://colab.research.google.com/drive/1d7cU0NsUj7jnOF1YSap_DH9r79G3-Cr4?usp=sharing#scrollTo=GqH4VGMroWec).

## Load a dataset
In this example we will work with supervisely-persons. If it's the first time you are using this dataset, or if you want to use another dataset please check out [dataset setup instructions](...)
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
    "lr_mode": "cosine",
    "initial_lr": 0.005,
    "lr_warmup_epochs": 5,
    "multiply_head_lr": 10,
    "optimizer": "SGD",
    "loss": "bce_dice_loss",
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
### How to launch on multiple GPUs (DDP) ?
Please check out our tutorial on [how to use multiple GPUs'](...)
