# Data in SG

To handle data, SuperGradients takes use of two Pytorch primitives- `torch.utils.data.Dataset` which is in charge of generating the samples and their corresponding labels,
and `torch.utils.data.DataLoader` that wraps an iterable around the Dataset to enable easy access to the samples. In other words, `torch.utils.data.Dataset` defines how to load a single sample,
while `torch.utils.data.DataLoader` defines how to load batches of samples. For more information see https://pytorch.org/docs/stable/data.html.

## SG Datasets

SuperGradients holds common public `torch.utils.data.Dataset` implementations for various tasks:

    Cifar10
    Cifar100
    ImageNetDataset
    COCODetectionDataset
    DetectionDataset
    PascalVOCDetectionDataset
    SegmentationDataSet
    CoCoSegmentationDataSet
    PascalAUG2012SegmentationDataSet
    PascalVOC2012SegmentationDataSet
    CityscapesDataset
    SuperviselyPersonsDataset
    PascalVOCAndAUGUnifiedDataset
    COCOKeypointsDataset

All of which can be imported from the `super_gradients.training.datasets` module. Note that some of the above implementations require following a few simple setup steps, which are all documented [here](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/datasets/Dataset_Setup_Instructions.md))
Once instantiated, any of the above can be simply passed to the `torch.utils.data.DataLoader` constructor, and be sued for training, validation or testing.

Creating a `torch.utils.data.DataLoader` from some dataset can be tricky, especially when needing to define some parameters on the fly. For example, in distributed training (i.e DDP)
the  `torch.utils.data.DataLoader` must be given a proper `Sampler`, such that the dataset indices will be divided among the different processes. Not acknowledging this and using the default
`torch.utils.data.SequentialSampler` will lead the different processes to iterate over the same data samples giving little to no speedup over single GPU training! User beware!
This is where SG's `training.dataloaders.get` comes in handy, by taking the burden of instantiating the right default sampler according to the training settings.

```python

from my_dataset import MyDataset
from super_gradients.training import dataloaders

dataset = MyDataset(transforms=T.ToTensor())
dataloader = dataloaders.get(dataset=dataset, dataloader_params={"batch_size": 4})
```

Note that `dataloader_params` will be unpacked in the `torch.utils.data.DataLoader` constructor, after setting a proper sampler if one is not explicitly set.

## SG DataLoaders

As mentioned above- once instantiated, the `torch.utils.data.DataLoader` objects are the ones forming batches.
Therefore- these are the objects being passed to Trainer.train(...):

```python
...
trainer = Trainer("my_experiment")
train_dataloader = ...
valid_dataloader = ...
model = ...
train_params = {...}

trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
   ```

For your convenience, SuperGradients gives full access to all data loader objects used in our training recipes.
These are simply the `torch.utils.data.DataLoader` configured by the recipe's `dataset_params`:

    cifar10_val
    cifar10_train
    cifar100_val
    cifar100_train
    coco2017_train
    coco2017_val
    coco2017_train_ssd_lite_mobilenet_v2
    coco2017_val_ssd_lite_mobilenet_v2
    imagenet_train
    imagenet_val
    imagenet_efficientnet_train
    imagenet_efficientnet_val
    imagenet_mobilenetv2_train
    imagenet_mobilenetv2_val
    imagenet_mobilenetv3_train
    imagenet_mobilenetv3_val
    imagenet_regnetY_train
    imagenet_regnetY_val
    imagenet_resnet50_train
    imagenet_resnet50_val
    imagenet_resnet50_kd_train
    imagenet_resnet50_kd_val
    imagenet_vit_base_train
    imagenet_vit_base_val
    tiny_imagenet_train
    tiny_imagenet_val
    pascal_aug_segmentation_train
    pascal_aug_segmentation_val
    pascal_voc_segmentation_train
    pascal_voc_segmentation_val
    supervisely_persons_train
    supervisely_persons_val
    pascal_voc_detection_train
    pascal_voc_detection_val

All of which can be imported from the `super_gradients.training.dataloaders` module, and 
Obviously, your training needs won't always align exactly with the same configuration as our recipes.
Therefore, overriding any underlying `Dataset` constructor parameter and any `DataLoader` parameter is possible through
the two named arguments: `dataset_params` and `dataloader_params`, which will override the recipe ones (by entry).


For example, the code below will instantiate the data loader used for training, in our `imagenet_resnet50` recipe
(including all data augmentations and any other data-related setting which we defined for training Resnet50 on Imagenet)
but changing the batch size for our needs.
We can then, also with a one-liner, instantiate the validation dataloader, and call train() as always:

```python
from super_gradients.training.dataloaders import imagenet_resnet50_train, imagenet_resnet50_val
from super_gradients.training import Trainer

train_dataloader = imagenet_resnet50_train(dataloader_params={"batch_size": 4})
valid_dataloader = imagenet_resnet50_val()

...
trainer = Trainer("my_imagenet_training_experiment")
model = ...
train_params = {...}

trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
   
   ```

### SG DataLoaders- Training with Configuration Files

If you are not familiar with training with configuration files, follow see this [link](https://github.com/Deci-AI/super-gradients/tree/master/documentation/source).

Any of the SG predefined data loaders listed earlier can be simply plugged in by their names.
For example, using the imagenet_resnet50_train and imagenet_resnet50_val:

```yaml

dataset_params: ...
...
train_dataloader: imagenet_resnet50_train
val_dataloader: imagenet_resnet50_val

...
```

Now, on the structure of `dataset_params`:
```yaml
train_dataset_params:
train_dataloader_params:
val_dataset_params:
val_dataloader_params:

```
As their names suggest- these entries `train_dataset_params` will be passed to the underlying `torch.utils.data.Dataset`
class constructor and `train_dataloader_params` will be passed to the `torch.utils.data.DataLoader` constructor.
As in the previous sub-section, both `train_dataloader_params` and `train_dataset_params` will override the the corresponding
parameters defined for the predefined data loader ( in our case, imagenet_resnet50 recipe's dataset_params.train_dataset_params, and
imagenet_renet50 recipe's dataset_params.train_dataloader_params).
This occurs equivalently for the validation set as well.
To demonstrate, to train with the same data settings as in the previous code snippet:
```yaml
train_dataloader: imagenet_resnet50_train
val_dataloader: imagenet_resnet50_val
dataset_params:
    train_dataset_params:
    train_dataloader_params:
      batch_size: 2
    val_dataset_params:
    val_dataloader_params:

```

Such config will result in training with `imagenet_resnet50` the exact same train data loader, but modifying the batch size to be 2, and using the same 
validation data loader as the original recipe.

## Using Custom Datasets in SG

Suppose we have already our own `torch.utils.data.Dataset` class:
```python
import torch

class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, image_size: int):
        ...
```

For coded training launch, we can simply instantiate it, then use it in the same way as the first code snippet to create
the data loaders and call train():


```python

from my_dataset import MyCustomDataset
from super_gradients.training import dataloaders, Trainer

train_dataset = MyCustomDataset(train=True, image_size=64)
valid_dataset = MyCustomDataset(train=False, image_size=128)
train_dataloader = dataloaders.get(dataset=train_dataset, dataloader_params={"batch_size": 4, "shuffle": True})
valid_dataloader = dataloaders.get(dataset=valid_dataset, dataloader_params={"batch_size": 16})

trainer = Trainer("my_custom_dataset_training_experiment")
model = ...
train_params = {...}

trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
   
```

### Using Custom Datasets in SG- Training with Configuration Files

When using configuration files, for example, training using train_from_recipe (or similar, when the underlying train method that is being called is Trainer.train_from_config(...)),  In your ``my_dataset.py``, register your dataset class by decorating the class with `register_dataset`:
```python
import torch
from super_gradients.common.registry.registry import register_dataset

@register_dataset("my_custom_dataset")
class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, image_size: int):
        ...
```
Then, use your newly registered dataset class in your configuration (of course, can be split, use defaults, etc), by specifying
plugging in in `dataset` entry, inside dataloader_params, while leaving out (or leaving empty) `train_dataloader` and `valid_dataloader`:


```yaml
dataset_params:
    train_dataset_params:
      train: True
      image_size: 64
    train_dataloader_params:
      dataset: my_custom_dataset
      batch_size: 4
      shuffle: True
    val_dataset_params:
      train: False
      image_size: 128
    val_dataloader_params:
      dataset: my_custom_dataset
      batch_size: 16

```

Last, in your ``my_train_from_recipe_script.py`` file, just import the newly registered class (even though the class itself is unused, just to trigger the registry):
        
```python

  from omegaconf import DictConfig
  import hydra
  import pkg_resources
  from my_dataset import MyCustomDataset
  from super_gradients import Trainer, init_trainer
  
  
  @hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
  def main(cfg: DictConfig) -> None:
      Trainer.train_from_config(cfg)
  
  
  def run():
      init_trainer()
      main()
  
  
  if __name__ == "__main__":
      run()
```
