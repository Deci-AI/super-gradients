# Data

To handle data, SuperGradients takes use of two Pytorch primitives: `torch.utils.data.Dataset` - which is in charge of generating the samples and their corresponding labels,
and `torch.utils.data.DataLoader` - that wraps an iterable around the Dataset to enable easy access to the samples. In other words, `torch.utils.data.Dataset` defines how to load a single sample,
while `torch.utils.data.DataLoader` defines how to load batches of samples. For more information, see [PyTorch documentation](https://pytorch.org/docs/stable/data.html).

## Datasets

SuperGradients holds common public `torch.utils.data.Dataset` implementations for various tasks:

    Classification:
        Cifar10
        Cifar100
        ImageNetDataset
    
    Object Detection:
        COCODetectionDataset
        DetectionDataset
        PascalVOCDetectionDataset
    
    Semantic Segmentation:
        CoCoSegmentationDataSet
        PascalAUG2012SegmentationDataSet
        PascalVOC2012SegmentationDataSet
        CityscapesDataset
        SuperviselyPersonsDataset
        PascalVOCAndAUGUnifiedDataset
    
    Pose Estimation:
        COCOKeypointsDataset

All of which can be imported from the `super_gradients.training.datasets` module. Note that some of the above implementations require following a few simple setup steps, which are all documented [here](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/datasets/Dataset_Setup_Instructions.md)

Creating a `torch.utils.data.DataLoader` from a dataset can be tricky, especially when defining some parameters on the fly. For example, in distributed training (i.e., DDP)
the  `torch.utils.data.DataLoader` must be given a proper `Sampler` such that the dataset indices will be divided among the different processes.
```
Warning: Using the wrong sampler when defining a data loader to be used with DDP will lead the different processes to iterate over the same data samples giving little to no speedup over single GPU training!
```
This is where SG's `training.dataloaders.get` comes in handy by taking the burden of instantiating the proper default sampler according to the training settings.
Once instantiated, any of the above can be passed to the `torch.utils.data.DataLoader` constructor and be used for training, validation, or testing:

```python

from my_dataset import MyDataset
from super_gradients.training import dataloaders
import torchvision.transforms as T
from super_gradients.training import Trainer
from super_gradients.training.metrics import Accuracy

trainer = Trainer("my_experiment")
train_dataset = MyDataset(split="train", transforms=T.ToTensor())
valid_dataset = MyDataset(split="validation", transforms=T.ToTensor())
test_dataset = MyDataset(split="test", transforms=T.ToTensor())

train_dataloader = dataloaders.get(dataset=train_dataset, dataloader_params={"batch_size": 4})
valid_dataloader = dataloaders.get(dataset=valid_dataset, dataloader_params={"batch_size": 16})
test_dataloader = dataloaders.get(dataset=test_dataset, dataloader_params={"batch_size": 16})

model = ...
train_params = {...}
trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)

trainer.test(model=trainer.net, test_loader=test_dataloader, test_metrics_list=[Accuracy()])
```

Note that `dataloader_params` will be unpacked in the `torch.utils.data.DataLoader` constructor after setting a proper sampler if one is not explicitly set.

## DataLoaders

As mentioned above, once instantiated, the `torch.utils.data.DataLoader` objects form batches.
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

These DataLoader can be imported from the super_gradients.training.dataloaders module.
Please note that these Dataset and DataLoader objects are already pre-defined with parameters required for specific training recipes. You can override these default parameters by passing two named arguments: dataset_params and dataloader_params(both of which are dictionaries), which will override the recipe settings. To learn which parameters you can override for each object, please refer to the YAML file with the same name.

For example, the code below will instantiate the data loader used for training in our `imagenet_resnet50` recipe
(including all data augmentations and any other data-related setting which we defined for training Resnet50 on Imagenet)
but changing the batch size for our needs.
We can then, also with a one-liner, instantiate the validation dataloader and call train() as always:

```python
from super_gradients.training.dataloaders import imagenet_resnet50_train, imagenet_resnet50_val
from super_gradients.training import Trainer

train_dataloader = imagenet_resnet50_train(dataloader_params={"batch_size": 4, "shuffle": True}, dataset_params={"root": "/my_data_dir/Imagenet/train"})
valid_dataloader = imagenet_resnet50_val(dataloader_params={"batch_size": 16}, dataset_params={"root": "/my_data_dir/Imagenet/val"})

...
trainer = Trainer("my_imagenet_training_experiment")
model = ...
train_params = {...}

trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```

### DataLoaders - Training with Configuration Files

If you are still getting familiar with training with configuration files, follow [this link](configuration_files.md).

Their names can reference any of the SG-predefined data loaders listed earlier.
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
As their names suggest- the parameters under `train_dataset_params` will be passed to the Dataset, and the parameters under `train_dataloader_params` will be given to the DataLoader.
As in the previous sub-section, both `train_dataloader_params` and `train_dataset_params` will override the corresponding
parameters defined for the predefined data loader ( in our case, imagenet_resnet50 recipe's dataset_params.train_dataset_params, and
imagenet_renet50 recipe's dataset_params.train_dataloader_params).
The same logic holds for the validation set as well.
To demonstrate, let's look at what a configuration for training with the same data settings as in the previous code snippet looks like:
```yaml
train_dataloader: imagenet_resnet50_train
val_dataloader: imagenet_resnet50_val
dataset_params:
    train_dataset_params:
      root: /my_data_dir/Imagenet/train
    train_dataloader_params:
      batch_size: 4
      shuffle: True
    val_dataset_params:
      root: /my_data_dir/Imagenet/val
    val_dataloader_params:
      batch_size: 16

```

### DataLoaders - Additional params

In addition to the parameters that are supported by the `torch.utils.data.DataLoader` class, SuperGradients also provide additional parameters:

* `min_samples` - When present, this parameter will guarantee that at least `min_samples` items will be processed in each epoch. It is useful when working with small datasets. 
To use this option, simply add this parameter to the `dataloader_params` dictionary, and set it to the desired value:
```yaml
train_dataloader: imagenet_resnet50_train
dataset_params:
    train_dataloader_params:
      batch_size: 4
      shuffle: True
      min_samples: 1024
```

On the technical side, when this parameter is se, SuperGradients will attach the RandomSampler to the DataLoader, and set it's  `num_samples` parameter to `min_samples`.

## Using Custom Datasets

Suppose we already have our own `torch.utils.data.Dataset` class:
```python
import torch

class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, image_size: int):
        ...

    def __getitem__(self, item):
        ...
        return inputs, targets # Or inputs, targets, additional_batch_items
```

#### A. `__getitem__`
You need to make sure that the `__getitem__` method of your dataset complies with the following format:
   - `inputs = batch_items[0]` : model input - The type might depend on the model you are using.
   - `targets = batch_items[1]` : Target that will be used to compute loss/metrics - The type might depend on the function you are using.
   - [OPTIONAL] `additional_batch_items = batch_items[2]` : Dict made of any additional item that you might want to use.

#### B. Train with your dataset
For coded training launch, we can instantiate it, then use it in the same way as the first code snippet to create
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

### Using Custom Datasets - Training with Configuration Files

When using configuration files, for example, training using train_from_recipe (or similar, when the underlying train method that is being called is Trainer.train_from_config(...)),  In your ``my_dataset.py``, register your dataset class by decorating the class with `register_dataset`:
```python
import torch
from super_gradients.common.registry.registry import register_dataset

@register_dataset("my_custom_dataset")
class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, image_size: int):
        ...
```
Then, use your newly registered dataset class in your configuration (of course, it can be split, use defaults, etc.) by referencing its name in the `dataset` entry inside dataloader_params while leaving out (or leaving empty) `train_dataloader` and `valid_dataloader`:


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

Last, in your ``my_train_from_recipe_script.py`` file, import the newly registered class (even though the class itself is unused, just to trigger the registry):
        
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
