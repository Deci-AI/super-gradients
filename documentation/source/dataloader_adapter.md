# Dataset Adapter

With diverse dataset structures available, ensuring compatibility with SuperGradients (SG) can be challenging. This is where the DataloaderAdapter plays a pivotal role. This tutorial takes you through the importance, implementation, and advantages of using the DataloaderAdapter with SG.

### Why Dataset Adapter?

Datasets come in a myriad of structures. However, SG requires data in a specific format.

For instance, consider the Object Detection Format:

Image format should be: (BS, H, W, C) i.e., channel last.
Targets should be in the format: (BS, 6), where 6 represents (sample_id, class_id, label, cx, cy, w, h).
The overhead of adjusting each dataset manually can be cumbersome. Enter DataloaderAdapter – designed to automatically understand your dataset structure and mold it for SG compatibility.


### Why Do We Need the Dataset Adapter?

While Datasets come in various structures and formats, SG expects data in a specific format to be able to run.


> Example: Object Detection Format
> - Image format: (BS, H, W, C) i.e. channel last
> - Targets format: (BS, 6) where 6 represents (sample_id, class_id, label, cx, > cy, w, h).


This means that you should either use one of SuperGradient's built-in Dataset class if it supports your dataset structure, or, if your dataset is too custom for it, inherit from SG datasets and bring all the required changes.

While this is all right in most cases, it can be cumbersome when you just want to quickly experiment with a new dataset.

To reduce this overhead, SuperGradients introduced the concept of `DataloaderAdapter`. Instead of requiring you to write all the transformations required to use SG, the `DataloaderAdapter` will infer anything possible directly from your data. Whenever something cannot be inferred with 100% confidence, you will be asked a question with all the required context for you to properly answer.

Let's see this in practice with an example. Let's start with `SBDataset` dataset

# Exemple 1 - Segmentation Adapter on `SBDataset` Dataset

In this section, we'll walk through the process of preparing the `SBDataset` dataset for use in SuperGradients. We'll highlight the challenges and demonstrate how the Adapter can simplify the process.


1. Preparing the Dataset without Adapter


```python
from torchvision.datasets import SBDataset

try:
  # There is a bug with `torchvision.datasets.SBDataset` that raises RuntimeError after downloading, so we just ignore it
  SBDataset(root="data", mode='segmentation', download=True)
except RuntimeError:
  pass
```

    Downloading https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz to data/benchmark.tgz


    100%|██████████| 1419539633/1419539633 [00:32<00:00, 43301819.66it/s]


    Extracting data/benchmark.tgz to data
    Downloading https://www.cs.cornell.edu/~bharathh/ to data/train_noval.txt


    20563it [00:00, 1012436.88it/s]



```python
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode


transforms = Compose([ToTensor(), Resize((512, 512), InterpolationMode.NEAREST)])
def sample_transform(image, mask):
  return transforms(image), transforms(mask)

train_set = SBDataset(root="data", mode='segmentation', download=False, transforms=sample_transform)
```

Now let's see what we get when instantiating a `Dataloader`


```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
_images, labels = next(iter(train_loader))

labels.unique()
```




    tensor([0.0000, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0353, 0.0431, 0.0471,
            0.0549, 0.0588, 0.0627, 0.0706, 0.0745, 0.0784])



As you can see, the labels are normalized (0-1). This is all right, but it is not the format expected by SuperGradients.

Let's now see how the Adapter helps.

2. Introducing Adapter

The Adapter helps us skip manual data preparations and dives right into creating a dataloader that SuperGradients expects.


```python
from super_gradients.training.dataloaders.adapters import SegmentationDataloaderAdapterFactory

train_loader = SegmentationDataloaderAdapterFactory.from_dataset(dataset=train_set, batch_size=20, shuffle=True, config_path='local_cache.json')

_images, labels = next(iter(train_loader))
labels.unique()
```

    [2023-10-29 15:25:36] INFO - data_config.py - Cache deactivated for `SegmentationDataConfig`.


    
    --------------------------------------------------------------------------------
    How many classes does your dataset include?
    --------------------------------------------------------------------------------
    
    Enter your response >>> 21
    Great! You chose: `21`
    
    --------------------------------------------------------------------------------
    Does your dataset provide a batch or a single sample?
    --------------------------------------------------------------------------------
        - Image shape: torch.Size([3, 512, 512])
        - Mask shape:  torch.Size([1, 512, 512])
    Options:
    [0] | Batch of Samples (e.g. torch Dataloader)
    [1] | Single Sample (e.g. torch Dataset)
    
    Your selection (Enter the corresponding number) >>> 1
    Great! You chose: `Single Sample (e.g. torch Dataset)`
    
    --------------------------------------------------------------------------------
    In which format are your images loaded ?
    --------------------------------------------------------------------------------
    
    Options:
    [0] | RGB
    [1] | BGR
    [2] | LAB
    [3] | Other
    
    Your selection (Enter the corresponding number) >>> 0
    Great! You chose: `RGB`





    tensor([ 0,  1,  2,  3,  4,  7,  8,  9, 12, 13, 15, 16, 18, 19, 20])



You can see that the mask is now encoded as `int`, which is the representation used in SuperGradients.

It's important to note that the dataset adapter also support different dataset format such as one hot, ensuring that the output (`labels` here) is in the right format to use within SuperGradients.

## Example II - Detection Adapter on a Dictionary based Dataset

Some datasets return a more complex data structure than the previous example.

For instance, the `COCO` dataset implementation from `pytorch` returns a list of dictionaries representing the labels.

Let's have a look:



```python
# Download the zip file
!wget https://deci-pretrained-models.s3.amazonaws.com/coco2017_small.zip

# Unzip the downloaded file
!unzip coco2017_small.zip > /dev/null
```

    --2023-10-29 15:27:31--  https://deci-pretrained-models.s3.amazonaws.com/coco2017_small.zip
    Resolving deci-pretrained-models.s3.amazonaws.com (deci-pretrained-models.s3.amazonaws.com)... 54.231.134.129, 52.217.71.68, 52.217.138.65, ...
    Connecting to deci-pretrained-models.s3.amazonaws.com (deci-pretrained-models.s3.amazonaws.com)|54.231.134.129|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 246116231 (235M) [application/zip]
    Saving to: ‘coco2017_small.zip’
    
    coco2017_small.zip  100%[===================>] 234.71M  38.7MB/s    in 6.7s    
    
    2023-10-29 15:27:38 (34.9 MB/s) - ‘coco2017_small.zip’ saved [246116231/246116231]
    



```python
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode
from torchvision.datasets import SBDataset


image_transform = Compose([ToTensor(), Resize((512, 512))])

train_set = CocoDetection(root='coco2017_small/images/train2017', annFile='coco2017_small/annotations/instances_train2017.json', transform=image_transform)
train_set = CocoDetection(root='coco2017_small/images/val2017', annFile='coco2017_small/annotations/instances_val2017.json', transform=image_transform)
image, targets = next(iter(train_set))
```

    loading annotations into memory...
    Done (t=0.10s)
    creating index...
    index created!
    loading annotations into memory...
    Done (t=0.05s)
    creating index...
    index created!



```python
print(f"Number of targets: {len(targets)}, First target structure: {targets[0]}")
```

Observe the dataset output's nested dictionary structure? This complicates things for the Dataset Adapter as it's unsure which fields detail the bounding box.

To solve this, we utilize an extractor function.

#### The Extractor's Role

Simply put, the extractor translates your dataset's output into a format the Adapter understands. For our dataset, it will take the image and annotations, then return the bounding box data, including the label and coordinates.

Worried about bounding box format like `xyxy_label` or `label_xywh`? Don't be. The Adapter is designed to recognize them.

> For further guidance on extractor functions, see the [official documentation](https://github.com/Deci-AI/data-gradients/blob/master/documentation/dataset_extractors.md).


```python
import torch

def coco_labels_extractor(sample) -> torch.Tensor:
    _, annotations = sample # annotations = [{"bbox": [1.08, 187.69, 611.59, 285.84], "category_id": 51}, ...]
    labels = []
    for annotation in annotations:
        class_id = annotation["category_id"]
        bbox = annotation["bbox"]
        labels.append((class_id, *bbox))
    return torch.Tensor(labels) # torch.Tensor([[51, 1.08, 187.69, 611.59, 285.84], ...])

coco_labels_extractor(sample=next(iter(train_set)))
```




    tensor([[ 64.0000, 236.9800, 142.5100,  24.7000,  69.5000],
            [ 72.0000,   7.0300, 167.7600, 149.3200,  94.8700],
            [ 72.0000, 557.2100, 209.1900,  81.3500,  78.7300],
            [ 62.0000, 358.9800, 218.0500,  56.0000, 102.8300],
            [ 62.0000, 290.6900, 218.0000,  61.8300,  98.4800],
            [ 62.0000, 413.2000, 223.0100,  30.1700,  81.3600],
            [ 62.0000, 317.4000, 219.2400,  21.5800,  11.5900],
            [  1.0000, 412.8000, 157.6100,  53.0500, 138.0100],
            [  1.0000, 384.4300, 172.2100,  15.1200,  35.7400],
            [ 78.0000, 512.2200, 205.7500,  14.7400,  15.9700],
            [ 82.0000, 493.1000, 174.3400,  20.2900, 108.3100],
            [ 84.0000, 604.7700, 305.8900,  14.3400,  45.7100],
            [ 84.0000, 613.2400, 308.2400,  12.8800,  46.4400],
            [ 85.0000, 447.7700, 121.1200,  13.9700,  21.8800],
            [ 86.0000, 549.0600, 309.4300,  36.6800,  89.6700],
            [ 86.0000, 350.7600, 208.8400,  11.3700,  22.5500],
            [ 62.0000, 412.2500, 219.0200,   9.6300,  12.5200],
            [ 86.0000, 241.2400, 194.9900,  14.2200,  17.6300],
            [ 86.0000, 336.7900, 199.5000,   9.7300,  16.7300],
            [ 67.0000, 321.2100, 231.2200, 125.5600,  88.9300]])



This output is all you need to get started. Now we can use the Dataloader Adapters!


```python
from super_gradients.training.dataloaders.adapters import DetectionDataloaderAdapterFactory
from data_gradients.dataset_adapters.config.data_config import DetectionDataConfig


adapter_config = DetectionDataConfig(labels_extractor=coco_labels_extractor, cache_path="coco_adapter_cache.json")
train_loader = DetectionDataloaderAdapterFactory.from_dataset(
    dataset=train_set,
    config=adapter_config,
    batch_size=5,
    drop_last=True,
)
val_loader = DetectionDataloaderAdapterFactory.from_dataset(
    dataset=train_set,
    config=adapter_config,
    batch_size=5,
    drop_last=True,
)
```

    /usr/local/lib/python3.10/dist-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
      warnings.warn(
    [2023-10-29 15:27:41] INFO - data_config.py - Cache deactivated for `DetectionDataConfig`.
    [2023-10-29 15:27:41] INFO - detection_adapter_collate_fn.py - You are using Detection Adapter. Please note that it was designed specifically for YOLONAS, YOLOX and PPYOLOE.


    Number of targets: 20, First target structure: {'segmentation': [[240.86, 211.31, 240.16, 197.19, 236.98, 192.26, 237.34, 187.67, 245.8, 188.02, 243.33, 176.02, 250.39, 186.96, 251.8, 166.85, 255.33, 142.51, 253.21, 190.49, 261.68, 183.08, 258.86, 191.2, 260.98, 206.37, 254.63, 199.66, 252.51, 201.78, 251.8, 212.01]], 'area': 531.8071000000001, 'iscrowd': 0, 'image_id': 139, 'bbox': [236.98, 142.51, 24.7, 69.5], 'category_id': 64, 'id': 26547}
    
    --------------------------------------------------------------------------------
    How many classes does your dataset include?
    --------------------------------------------------------------------------------
    
    Enter your response >>> 80
    Great! You chose: `80`
    
    --------------------------------------------------------------------------------
    In which format are your images loaded ?
    --------------------------------------------------------------------------------
    
    Options:
    [0] | RGB
    [1] | BGR
    [2] | LAB
    [3] | Other
    
    Your selection (Enter the corresponding number) >>> 0
    Great! You chose: `RGB`
    
    --------------------------------------------------------------------------------
    Which comes first in your annotations, the class id or the bounding box?
    --------------------------------------------------------------------------------
    Here's a sample of how your labels look like:
    Each line corresponds to a bounding box.
    tensor([[ 64.0000, 236.9800, 142.5100,  24.7000,  69.5000],
            [ 72.0000,   7.0300, 167.7600, 149.3200,  94.8700],
            [ 72.0000, 557.2100, 209.1900,  81.3500,  78.7300],
            [ 62.0000, 358.9800, 218.0500,  56.0000, 102.8300]])
    Options:
    [0] | Label comes first (e.g. [class_id, x1, y1, x2, y2])
    [1] | Bounding box comes first (e.g. [x1, y1, x2, y2, class_id])
    
    Your selection (Enter the corresponding number) >>> 0
    Great! You chose: `Label comes first (e.g. [class_id, x1, y1, x2, y2])`
    
    --------------------------------------------------------------------------------
    What is the bounding box format?
    --------------------------------------------------------------------------------
    Here's a sample of how your labels look like:
    Each line corresponds to a bounding box.
    tensor([[ 64.0000, 236.9800, 142.5100,  24.7000,  69.5000],
            [ 72.0000,   7.0300, 167.7600, 149.3200,  94.8700],
            [ 72.0000, 557.2100, 209.1900,  81.3500,  78.7300],
            [ 62.0000, 358.9800, 218.0500,  56.0000, 102.8300]])
    Options:
    [0] | xyxy: x-left, y-top, x-right, y-bottom		(Pascal-VOC format)
    [1] | xywh: x-left, y-top, width, height			(COCO format)
    [2] | cxcywh: x-center, y-center, width, height		(YOLO format)
    
    Your selection (Enter the corresponding number) >>> 1


    [2023-10-29 15:28:40] INFO - detection_adapter_collate_fn.py - You are using Detection Adapter. Please note that it was designed specifically for YOLONAS, YOLOX and PPYOLOE.



```python
_image, targets = next(iter(train_loader))
```


```python
targets.shape # [N, 6] format with 6 representing (sample_id, class_id, cx, cy, w, h)
```




    torch.Size([22, 6])




```python
targets[:3]
```




    tensor([[  0.0000,  64.0000, 249.3300, 177.2600,  24.7000,  69.5000],
            [  0.0000,  72.0000,  81.6900, 215.1950, 149.3200,  94.8700],
            [  0.0000,  72.0000, 597.8850, 248.5550,  81.3500,  78.7300]])



# III. Use your Adapted Dataloader to train a model

Now that we have an adapter for a detection dataset, let's use it to launch a training of YoloNAS on it!

This is of course for the sake of the example, since YoloNAS was originally trained using the SuperGradients implementation of COCO Dataset. You can replace the `COCO` dataset with any of your dataset.

```python
from omegaconf import OmegaConf
from hydra.utils import instantiate

from super_gradients import Trainer
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training import training_hyperparams
from super_gradients.common.environment.cfg_utils import load_recipe

trainer = Trainer(experiment_name="yolonas_training_with_adapter", ckpt_root_dir="../../scripts/")
model = models.get(model_name=Models.YOLO_NAS_S, num_classes=adapter_config.n_classes, pretrained_weights="coco")

yolonas_recipe = load_recipe(config_name="coco2017_yolo_nas_s",
                             overrides=[f"arch_params.num_classes={adapter_config.n_classes}",
                                        "training_hyperparams.max_epochs=1",
                                        "training_hyperparams.mixed_precision=False"])
yolonas_recipe = OmegaConf.to_container(instantiate(yolonas_recipe))
training_params = yolonas_recipe['training_hyperparams']

trainer.train(model=model, training_params=training_params, train_loader=train_loader, valid_loader=val_loader)
```

    [2023-10-29 15:29:42] INFO - checkpoint_utils.py - License Notification: YOLO-NAS pre-trained weights are subjected to the specific license terms and conditions detailed in 
    https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md
    By downloading the pre-trained weight files you agree to comply with these terms.
    Downloading: "https://sghub.deci.ai/models/yolo_nas_s_coco.pth" to /root/.cache/torch/hub/checkpoints/yolo_nas_s_coco.pth
    100%|██████████| 73.1M/73.1M [00:00<00:00, 81.0MB/s]
    [2023-10-29 15:29:43] INFO - checkpoint_utils.py - Successfully loaded pretrained weights for architecture yolo_nas_s
    [2023-10-29 15:29:44] INFO - sg_trainer.py - Starting a new run with `run_id=RUN_20231029_152944_310569`
    [2023-10-29 15:29:44] INFO - sg_trainer.py - Checkpoints directory: ./yolonas_training_with_adapter/RUN_20231029_152944_310569
    [2023-10-29 15:29:44] INFO - sg_trainer.py - Using EMA with params {'decay': 0.9997, 'decay_type': 'threshold', 'beta': 15}


    The console stream is now moved to ./yolonas_training_with_adapter/RUN_20231029_152944_310569/console_Oct29_15_29_44.txt


    [2023-10-29 15:29:45] WARNING - callbacks.py - Number of warmup steps (1000) is greater than number of steps in epoch (100). Warmup steps will be capped to number of steps in epoch to avoid interfering with any pre-epoch LR schedulers.
    /usr/local/lib/python3.10/dist-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/collate_fn/detection_collate_fn.py:29: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      images_batch = [torch.tensor(img) for img in images_batch]
    /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/collate_fn/detection_collate_fn.py:43: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      labels_batch = [torch.tensor(labels) for labels in labels_batch]
    [2023-10-29 15:29:45] INFO - sg_trainer_utils.py - TRAINING PARAMETERS:
        - Mode:                         Single GPU
        - Number of GPUs:               0          (0 available on the machine)
        - Full dataset size:            500        (len(train_set))
        - Batch size per GPU:           5          (batch_size)
        - Batch Accumulate:             1          (batch_accumulate)
        - Total batch size:             5          (num_gpus * batch_size)
        - Effective Batch size:         5          (num_gpus * batch_size * batch_accumulate)
        - Iterations per epoch:         100        (len(train_loader))
        - Gradient updates per epoch:   100        (len(train_loader) / batch_accumulate)
    
    [2023-10-29 15:29:45] INFO - sg_trainer.py - Started training for 1 epochs (0/0)
    
    Train epoch 0: 100%|██████████| 100/100 [20:18<00:00, 12.18s/it, PPYoloELoss/loss=4, PPYoloELoss/loss_cls=1.72, PPYoloELoss/loss_dfl=2.2, PPYoloELoss/loss_iou=0.472, gpu_mem=0]
    Validating: 100%|██████████| 100/100 [06:34<00:00,  3.94s/it]
    [2023-10-29 15:56:43] INFO - base_sg_logger.py - Checkpoint saved in ./yolonas_training_with_adapter/RUN_20231029_152944_310569/ckpt_best.pth
    [2023-10-29 15:56:43] INFO - sg_trainer.py - Best checkpoint overriden: validation mAP@0.50:0.95: 0.0005365016404539347


    ===========================================================
    SUMMARY OF EPOCH 0
    ├── Train
    │   ├── Ppyoloeloss/loss_cls = 1.7168
    │   ├── Ppyoloeloss/loss_iou = 0.4717
    │   ├── Ppyoloeloss/loss_dfl = 2.2035
    │   └── Ppyoloeloss/loss = 3.9977
    └── Validation
        ├── Ppyoloeloss/loss_cls = 2.4528
        ├── Ppyoloeloss/loss_iou = 0.5016
        ├── Ppyoloeloss/loss_dfl = 2.2003
        ├── Ppyoloeloss/loss = 4.807
        ├── Precision@0.50:0.95 = 0.0052
        ├── Recall@0.50:0.95 = 0.007
        ├── Map@0.50:0.95 = 0.0005
        └── F1@0.50:0.95 = 0.0007
    
    ===========================================================


    [2023-10-29 15:56:45] INFO - sg_trainer.py - RUNNING ADDITIONAL TEST ON THE AVERAGED MODEL...
    Validating epoch 1: 100%|██████████| 100/100 [06:33<00:00,  3.93s/it]


# IV. Dig deeper into the Adapter

By default, any parameter that could not be confidently infered will trigger a question.

But you have the possibility to set these parameters in advance through the config object. In the previous example we had to set `labels_extractor` explicitly. Now let's set all the parameters


```python
from super_gradients.training.dataloaders.adapters import DetectionDataloaderAdapterFactory
from data_gradients.dataset_adapters.config.data_config import DetectionDataConfig
from data_gradients.utils.data_classes.image_channels import ImageChannels
class_names = [category['name'] for category in train_set.coco.loadCats(train_set.coco.getCatIds())]

adapter_config = DetectionDataConfig(
    labels_extractor=coco_labels_extractor,
    is_label_first=True,
    class_names=class_names,
    image_channels=ImageChannels.from_str("RGB"),
    xyxy_converter='xywh',
    cache_path="coco_adapter_cache_with_default.json"
)
```

This can now be used and you don't need to answer any question


```python
train_loader = DetectionDataloaderAdapterFactory.from_dataset(
    dataset=train_set,
    config=adapter_config,
    batch_size=5,
    drop_last=True,
)
val_loader = DetectionDataloaderAdapterFactory.from_dataset(
    dataset=train_set,
    config=adapter_config,
    batch_size=5,
    drop_last=True,
)

_image, targets = next(iter(train_loader))

print(targets.shape) # [N, 6] format with 6 representing (sample_id, class_id, cx, cy, w, h)
```

    [2023-10-29 16:15:09] INFO - detection_adapter_collate_fn.py - You are using Detection Adapter. Please note that it was designed specifically for YOLONAS, YOLOX and PPYOLOE.
    /usr/local/lib/python3.10/dist-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
      warnings.warn(
    [2023-10-29 16:15:09] INFO - detection_adapter_collate_fn.py - You are using Detection Adapter. Please note that it was designed specifically for YOLONAS, YOLOX and PPYOLOE.
    /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/collate_fn/detection_collate_fn.py:29: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      images_batch = [torch.tensor(img) for img in images_batch]
    /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/collate_fn/detection_collate_fn.py:43: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      labels_batch = [torch.tensor(labels) for labels in labels_batch]


### Load from existing cache

You can use the cache of an adapter you already used in the past. This will allow you skip the questions that were already asked in the previous run.


```python
# The new config will load the answer from questions asked in the previous run.
adapter_config = DetectionDataConfig(
    labels_extractor=coco_labels_extractor,
    cache_path="coco_adapter_cache_with_default.json" # Name of the previous cache
)

train_loader = DetectionDataloaderAdapterFactory.from_dataset(
    dataset=train_set,
    config=adapter_config,
    batch_size=5,
    drop_last=True,
)
val_loader = DetectionDataloaderAdapterFactory.from_dataset(
    dataset=train_set,
    config=adapter_config,
    batch_size=5,
    drop_last=True,
)

_image, targets = next(iter(train_loader))
```


```python
targets.shape # [N, 6] format with 6 representing (sample_id, class_id, cx, cy, w, h)
```




    torch.Size([22, 6])



As you can see, no question was asked and we still get the targets adapted into the SuperGradients format.
