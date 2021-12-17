<div align="center">
  <img src="https://github.com/Deci-AI/super-gradients/SG_img/SG - Horizontal.png" width="600"/>
</div>

# SuperGradients

## Introduction

Welcome to SuperGradients, a free open-source training library for PyTorch-based deep learning models.
There are two ways you can install it on your local machine - using this GitHub repository or using SuperGradients' private PyPi
repository.
The library lets you train models from any Computer Vision tasks or import pre-trained SOTA models, such as object detection, classification of images, and semantic segmentation for videos or images use cases.

Whether you are a beginner or an expert it is likely that you already have your own training script, model, loss function implementation etc.
In this notebook we present the modifications needed in order to launch your training so you can benefit from the various tools the SuperGradients has to offer.
## "Wait, but what's in it for me?"

Great question! our short answer is - Easy to use SOTA DL training library.

Our long answer - 

*   Train models from any Computer Vision tasks or import [pre-trained SOTA models](https://github.com/Deci-AI/super-gradients#pretrained-classification-pytorch-checkpoints) (detection, segmentation, and classification - YOLOv5, DDRNet, EfficientNet, RegNet, ResNet, MobileNet, etc.)
*  Shorten the training process using tested and proven [recipes](https://github.com/Deci-AI/super-gradients/tree/master/recipes) & [code examples](https://github.com/Deci-AI/super-gradients/tree/master/examples)
*  Easily configure your own or  use plug&play training, dataset , and architecture parameters.
*  Save time and easily integrate it into your codebase.


Table of Content:
<!-- toc -->

- [Quick Start Notebook](#quick-start-notebook)
- [Walkthrough Notebook](#supergradients-walkthrough-notebook)
- [Installation Methods](#installation-methods)
    - [Quick Installation of stable version](#quick-installation-of-stable-version)
    - [Installing from GitHub](#installing-from-github)
    - [Installing from AWS Codeartifact PyPi repository](#installing-from-aws-codeartifact-pypi-repository)
- [Computer Vision Models' Pretrained Checkpoints](#computer-vision-models-pretrained-checkpoints)
  - [Pretrained Classification PyTorch Checkpoints](#pretrained-classification-pytorch-checkpoints)
  - [Pretrained Object Detection PyTorch Checkpoints](#pretrained-object-detection-pytorch-checkpoints)
  - [Pretrained Semantic Segmentation PyTorch Checkpoints](#pretrained-semantic-segmentation-pytorch-checkpoints)
- [Development Flow](#development-flow)
    - [Feature and bugfix branches](#feature-and-bugfix-branches)
    - [Merging to Master](#merging-to-master)
    - [Creating a release](#creating-a-release)
- [Technical Debt](#technical-debt)

<!-- tocstop -->


## Quick Start Notebook

Get started with our quick start notebook on Google Colab for a quick and easy start using free GPU hardware

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://colab.research.google.com/drive/1lole-odbkD4LBnM6debK31BLP5_aILwY?usp=sharing"><img src="https://github.com/Deci-AI/super-gradients/SG_img/colab_logo.png" />SuperGradients Quick Start in Google Colab</a>
 </td>
  <td>
   <a href="https://github.com/Deci-AI/super-gradients/blob/master/examples/quickstart.ipynb"><img src="https://github.com/Deci-AI/super-gradients/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tutorials"><img src="https://github.com/Deci-AI/super-gradients/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>
 


## SuperGradients Walkthrough Notebook

Learn more about SuperGradients training components with our walkthrough notebook on Google Colab for an easy to use tutorial using free GPU hardware

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://colab.research.google.com/drive/1smwh4EAgE8PwnCtwsdU8a9D9Ezfh6FQK?usp=sharing"><img src="https://github.com/Deci-AI/super-gradients/SG_img/colab_logo.png" />SuperGradients Walkthrough in Google Colab</a>
 </td>
  <td>
   <a href="https://github.com/Deci-AI/super-gradients/blob/master/examples/SG_Walkthrough.ipynb"><img src="https://github.com/Deci-AI/super-gradients/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tutorials"><img src="https://github.com/Deci-AI/super-gradients/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>


## Installations

### Quick Installation of stable version
**Not yet avilable in PyPi**
```bash
  pip install super-gradients
```

That's it !

### Installing from GitHub
```bash
pip install git+https://github.com/Deci-AI/super-gradients.git@stable
```


## Computer Vision Models' Pretrained Checkpoints 

### Pretrained Classification PyTorch Checkpoints

##### **  TODO - ADD HERE EFFICIENCY FRONTIER CLASSIFICATION MODELS GRAPH FOR LATENCY **

| Model | Dataset |  Resolution |    Top-1    |    Top-5   | Latency b1<sub>T4</sub> | Throughout b1<sub>T4</sub> | 
|-------------------- |------ | ---------- |----------- |------ | -------- |  :------: |
| EfficientNet B0 | ImageNet  |224x224   |  77.62   | 93.49  |**1.16ms** |**862fps** |
| RegNetY200 | ImageNet  |224x224   |  70.88    |   89.35  |**-**|**-** |
| RegNetY400  | ImageNet  |224x224   |  74.74    |   91.46  |**-** |**-** |
| RegNetY600  | ImageNet  |224x224   |  76.18    |  92.34   |**-** |**-** |
| RegNetY800   | ImageNet  |224x224   |  77.07    |  93.26   |**-** |**-** |
| ResNet18   | ImageNet  |224x224   |  70.6    |   89.64 |**0.599ms** |**1669fps** |
| ResNet34  | ImageNet  |224x224   |  74.13   |   91.7  |**0.89ms** |**1123fps** |
| ResNet50  | ImageNet  |224x224   |  76.3    |   93.0  |**0.94ms** |**1063fps** |
| MobileNetV3_large-150 epochs | ImageNet  |224x224   |  73.79    |   91.54  |**0.87ms** |**1149fps** |
| MobileNetV3_large-300 epochs  | ImageNet  |224x224   |  74.52    |  91.92 |**0.87ms** |**1149fps** |
| MobileNetV3_small | ImageNet  |224x224   |67.45    |  87.47   |**0.75ms** |**1333fps** |
| MobileNetV2_w1   | ImageNet  |224x224   |  73.08 | 91.1  |**0.58ms** |**1724fps** |



### Pretrained Object Detection PyTorch Checkpoints

##### ** TODO - ADD HERE THE EFFICIENCY FRONTIER OBJECT-DETECTION MODELS GRAPH FOR LATENCY **


| Model | Dataset |  Resolution | mAP<sup>val<br>0.5:0.95 | Latency b1<sub>T4</sub> | Throughout b64<sub>T4</sub>  |
|--------------------- |------ | ---------- |------ | -------- |   :------: |
| YOLOv5 small | CoCo |640x640 |37.3   |**10.09ms** |**101.85fps** |
| YOLOv5 medium  | CoCo |640x640 |45.2   |**17.55ms** |**57.66fps** |


### Pretrained Semantic Segmentation PyTorch Checkpoints

##### ** TODO - ADD HERE THE EFFICIENCY FRONTIER SEMANTIC-SEGMENTATION MODELS GRAPH FOR LATENCY **


| Model | Dataset |  Resolution | mIoU | Latency<sub>T4</sub> | Throughout<sub>T4</sub>  |
|--------------------- |------ | ---------- | ------ | -------- | :------: |
| DDRNet23   | Cityscapes |1024x2048      |78.65     |**-** |**-** |
| DDRNet23 slim   | Cityscapes |1024x2048 |76.6    |**-** |**-** |



## Contributing

To learn about making a contribution to SuperGradients, please see our [Contribution page](CONTRIBUTING.md).

## Citation

If you use SuperGradients library or benchmark in your research, please cite SuperGradients deep learning training library.
    
## Community

If you want to be a part of SuperGradients growing community, hear about all the exciting news and updates, need help, request for advanced features, 
    or want to file a bug or issue report, we would love to welcome you aboard!

* [Slack](https://) is the place to be and ask questions about SuperGradients and get support. [Click here to join our Slack](
  https://).
* To report a bug, [file an issue](https://github.com/Deci-AI/super-gradients/issues) on GitHub.
* You can also join the [community mailing list](https://)
  to ask questions about the project and receive announcements.

## License

This project is released under the [Apache 2.0 license](LICENSE).
    
