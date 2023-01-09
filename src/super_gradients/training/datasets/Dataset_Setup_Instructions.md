## Computer Vision Datasets Setup

SuperGradients provides multiple Datasets implementations.

### Classification Datasets

<details>
<summary>Cifar10</summary>
 
Supports `download`

```python
from super_gradients.training.datasets import Cifar10
dataset = Cifar10(..., download=True)
```
</details>


<details>
<summary>Imagenet</summary>
1. Download imagenet dataset:
   - https://image-net.org/download.php
2. Unzip:
```
    Imagenet
     ├──train
     │  ├──n02093991
     │  │   ├──n02093991_1001.JPEG
     │  │   ├──n02093991_1004.JPEG
     │  │   └──...
     │  ├──n02093992
     │  └──...
     └──val
        ├──n02093991
        ├──n02093992
        └──...
```

3. Instantiate the dataset:
```python
from super_gradients.training.datasets import ImageNetDataset
train_set = ImageNetDataset(root='.../Imagenet/train', ...)
valid_set = ImageNetDataset(root='.../Imagenet/val', ...)
```
</details>


### Detection Datasets

<details>
<summary>Coco</summary>

1. Download coco dataset:
    - annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    - train2017: http://images.cocodataset.org/zips/train2017.zip
    - val2017: http://images.cocodataset.org/zips/val2017.zip

2. Unzip and organize it as below:
```
    coco
    ├── annotations
    │      ├─ instances_train2017.json
    │      ├─ instances_val2017.json
    │      └─ ...
    └── images
        ├── train2017
        │   ├─ 000000000001.jpg
        │   └─ ...
        └── val2017
            └─ ...
```

3. Instantiate the dataset:
```python
from super_gradients.training.datasets import COCODetectionDataset
train_set = COCODetectionDataset(data_dir='.../coco', subdir='images/train2017', json_file='instances_train2017.json', ...)
valid_set = COCODetectionDataset(data_dir='.../coco', subdir='images/val2017', json_file='instances_val2017.json', ...)
```
</details>


<details>
<summary>PascalVOC 2007 & 2012</summary>

Supports `download`
```python
from super_gradients.training.datasets import PascalVOCDetectionDataset
train_set = PascalVOCDetectionDataset(download=True, ...)
```

Dataset Structure:
```
    Dataset structure:
        ├─images
        │   ├─ train2012
        │   ├─ val2012
        │   ├─ VOCdevkit
        │   │    ├─ VOC2007
        │   │    │  ├──JPEGImages
        │   │    │  ├──SegmentationClass
        │   │    │  ├──ImageSets
        │   │    │  ├──ImageSets/Segmentation
        │   │    │  ├──ImageSets/Main
        │   │    │  ├──ImageSets/Layout
        │   │    │  ├──Annotations
        │   │    │  └──SegmentationObject
        │   │    └──VOC2012
        │   │       ├──JPEGImages
        │   │       ├──SegmentationClass
        │   │       ├──ImageSets
        │   │       ├──ImageSets/Segmentation
        │   │       ├──ImageSets/Main
        │   │       ├──ImageSets/Action
        │   │       ├──ImageSets/Layout
        │   │       ├──Annotations
        │   │       └──SegmentationObject
        │   ├─train2007
        │   ├─test2007
        │   └─val2007
        └─labels
            ├─train2012
            ├─val2012
            ├─train2007
            ├─test2007
            └─val2007
```
</details>



### Segmentation Datasets

<details>
<summary>Coco</summary>

1. Download coco dataset:
    - annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    - train2017: http://images.cocodataset.org/zips/train2017.zip
    - val2017: http://images.cocodataset.org/zips/val2017.zip

2. Unzip and organize it as below:
```
    coco
    ├── annotations
    │      ├─ instances_train2017.json
    │      ├─ instances_val2017.json
    │      └─ ...
    └── images
        ├── train2017
        │   ├─ 000000000001.jpg
        │   └─ ...
        └── val2017
            └─ ...
```

3. Instantiate the dataset:
```python
from super_gradients.training.datasets import CoCoSegmentationDataSet
train_set = CoCoSegmentationDataSet(data_dir='.../coco', subdir='images/train2017', json_file='instances_train2017.json', ...)
valid_set = CoCoSegmentationDataSet(data_dir='.../coco', subdir='images/val2017', json_file='instances_val2017.json', ...)
```
</details>


<details>
<summary>Pascal VOC 2012</summary>

1. Download pascal datasets:
   - VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

2. Unzip and organize it as below:
```
   pascal_voc_2012
       └──VOCdevkit
             └──VOC2012
                ├──JPEGImages
                ├──SegmentationClass
                ├──ImageSets
                │    ├──Segmentation
                │    │   └── train.txt
                │    ├──Main
                │    ├──Action
                │    └──Layout
                ├──Annotations
                └──SegmentationObject
```

3. Instantiate the dataset:
```python
from super_gradients.training.datasets import PascalVOC2012SegmentationDataSet

train_set = PascalVOC2012SegmentationDataSet(
     root='.../pascal_voc_2012',
     list_file='VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
     samples_sub_directory='VOCdevkit/VOC2012/JPEGImages',
     targets_sub_directory='VOCdevkit/VOC2012/SegmentationClass',
     ...
 )
valid_set = PascalVOC2012SegmentationDataSet(
     root='.../pascal_voc_2012',
     list_file='VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
     samples_sub_directory='VOCdevkit/VOC2012/JPEGImages',
     targets_sub_directory='VOCdevkit/VOC2012/SegmentationClass',
     ...
 )
```
</details>


<details>
<summary>Pascal AUG 2012</summary>

1. Download pascal datasets:
   - AUG 2012: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

2. Unzip and organize it as below:
```
   pascal_voc_2012
       └──VOCaug
           ├── aug.txt
           └── dataset
                 ├──inst
                 ├──img
                 └──cls
```


3. Instantiate the dataset:
```python
from super_gradients.training.datasets import PascalAUG2012SegmentationDataSet

train_set = PascalAUG2012SegmentationDataSet(
     root='.../pascal_voc_2012',
     list_file='VOCaug/dataset/aug.txt',
     samples_sub_directory='VOCaug/dataset/img',
     targets_sub_directory='VOCaug/dataset/cls',
     ...
 )
```

NOTE: this dataset is only available for training. To test, please use PascalVOC2012SegmentationDataSet.
 </details>



<details>
<summary>Pascal AUG & VOC 2012</summary>

1. Download pascal datasets:
   - VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
   - AUG 2012: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

2. Unzip and organize it as below:
```
   pascal_voc_2012
       ├─VOCdevkit
       │ └──VOC2012
       │    ├──JPEGImages
       │    ├──SegmentationClass
       │    ├──ImageSets
       │    │    ├──Segmentation
       │    │    │   └── train.txt
       │    │    ├──Main
       │    │    ├──Action
       │    │    └──Layout
       │    ├──Annotations
       │    └──SegmentationObject
       └──VOCaug
           ├── aug.txt
           └── dataset
                 ├──inst
                 ├──img
                 └──cls
```


3. Instantiate the dataset:
```python
from super_gradients.training.datasets import PascalVOCAndAUGUnifiedDataset
train_set = PascalVOCAndAUGUnifiedDataset(root='.../pascal_voc_2012', ...)
```

 NOTE: this dataset is only available for training. To test, please use PascalVOC2012SegmentationDataSet.
 </details>


<details>
<summary>Supervisely Persons</summary>

1. Download supervisely dataset:
   - https://deci-pretrained-models.s3.amazonaws.com/supervisely-persons.zip)

2. Unzip:
```
   supervisely-persons
    ├──images
    │    ├──image-name.png
    │    └──...
    ├──images_600x800
    │    ├──image-name.png
    │    └──...
    ├──masks
    └──masks_600x800
```


3. Instantiate the dataset:
```python
from super_gradients.training.datasets import SuperviselyPersonsDataset
train_set = SuperviselyPersonsDataset(root_dir='.../supervisely-persons', list_file='train.csv', ...)
valid_set = SuperviselyPersonsDataset(root_dir='.../supervisely-persons', list_file='val.csv', ...)
```

NOTE: this dataset is only available for training. To test, please use PascalVOC2012SegmentationDataSet.
 </details>
