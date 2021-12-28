<div align="center">
  <img src="./docs/assets/SG_img/SG - Horizontal.png" width="600"/>
</div>

# SuperGradients

## Introduction

Welcome to SuperGradients, a free open-source training library for PyTorch-based deep learning models. SuperGradients allows you to train models of any computer vision tasks or import pre-trained SOTA models, such as object detection, classification of images, and semantic segmentation for videos and images.

Whether you are a beginner or an expert it is likely that you already have your own training script, model, loss function implementation, etc., and thus you experienced with how difficult it is to develop a production ready deep learning model, the overhead of integrating with existing training tools with very different and stiff formats and conventions, how much effort it is to find a suitable architecture for your needs when every repo is focusing on just one task.

With SuperGradients you can:

*   Train models for any Computer Vision task or import production-ready [pre-trained SOTA models](https://github.com/Deci-AI/super-gradients#pretrained-classification-pytorch-checkpoints) (detection, segmentation, and classification - YOLOv5, DDRNet, EfficientNet, RegNet, ResNet, MobileNet, etc.)
*  Shorten the training process using tested and proven [recipes](https://github.com/Deci-AI/super-gradients/tree/master/recipes) & [code examples](https://github.com/Deci-AI/super-gradients/tree/master/examples)
*  Easily configure your own or use plug&play training, dataset, and architecture parameters.
*  Save time and easily integrate it into your codebase.


### Table of Content:
<!-- toc -->

- [Getting Started](#getting-started)
    - [Quick Start Notebook](#quick-start-notebook)
    - [Walkthrough Notebook](#supergradients-walkthrough-notebook)
- [Installation Methods](#installation-methods)
    - [Prerequisites](#prerequisites)
    - [Quick Installation of stable version](#quick-installation-of-stable-version)
    - [Installing from GitHub](#installing-from-github)
- [Computer Vision Models' Pretrained Checkpoints](#computer-vision-models-pretrained-checkpoints)
  - [Pretrained Classification PyTorch Checkpoints](#pretrained-classification-pytorch-checkpoints)
  - [Pretrained Object Detection PyTorch Checkpoints](#pretrained-object-detection-pytorch-checkpoints)
  - [Pretrained Semantic Segmentation PyTorch Checkpoints](#pretrained-semantic-segmentation-pytorch-checkpoints)
- [Contributing](#contributing)
- [Citation](#citation)
- [Community](#community)
- [License](#license)

<!-- tocstop -->

## Getting Started

### Quick Start Notebook

Get started with our quick start notebook on Google Colab for a quick and easy start using free GPU hardware

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://colab.research.google.com/drive/12cURMPVQrvhgYle-wGmE2z8b_p90BdL0?usp=sharing"><img src="./docs/assets/SG_img/colab_logo.png" />SuperGradients Quick Start in Google Colab</a>
 </td>
  <td>
   <a href="https://github.com/Deci-AI/super-gradients/blob/master/examples/quickstart.ipynb"><img src="./docs/assets/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tutorials"><img src="./docs/assets/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>



### SuperGradients Walkthrough Notebook

Learn more about SuperGradients training components with our walkthrough notebook on Google Colab for an easy to use tutorial using free GPU hardware

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://colab.research.google.com/drive/1smwh4EAgE8PwnCtwsdU8a9D9Ezfh6FQK?usp=sharing"><img src="./docs/assets/SG_img/colab_logo.png" />SuperGradients Walkthrough in Google Colab</a>
 </td>
  <td>
   <a href="https://github.com/Deci-AI/super-gradients/blob/master/examples/SG_Walkthrough.ipynb"><img src="./docs/assets/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tutorials"><img src="./docs/assets/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>


## Installation Methods
### Prerequisites
General requirements:
- Python 3.7, 3.8 or 3.9 installed.
- torch>=1.9.0
- requirements.txt

To train on nvidia GPUs:
- [Nvidia CUDA Toolkit >= 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu)
- CuDNN >= 8.1.x
- Nvidia Driver with CUDA >= 11.2 support (≥460.x)

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
| RegNetY200 | ImageNet  |224x224   |  70.88    |   89.35  |**1.07ms**|**928.3fps** |
| RegNetY400  | ImageNet  |224x224   |  74.74    |   91.46  |**1.22ms** |**816.5fps** |
| RegNetY600  | ImageNet  |224x224   |  76.18    |  92.34   |**1.19ms** |**838.5fps** |
| RegNetY800   | ImageNet  |224x224   |  77.07    |  93.26   |**1.18ms** |**841.4fps** |
| ResNet18   | ImageNet  |224x224   |  70.6    |   89.64 |**0.599ms** |**1669fps** |
| ResNet34  | ImageNet  |224x224   |  74.13   |   91.7  |**0.89ms** |**1123fps** |
| ResNet50  | ImageNet  |224x224   |  76.3    |   93.0  |**0.94ms** |**1063fps** |
| MobileNetV3_large-150 epochs | ImageNet  |224x224   |  73.79    |   91.54  |**0.87ms** |**1149fps** |
| MobileNetV3_large-300 epochs  | ImageNet  |224x224   |  74.52    |  91.92 |**0.87ms** |**1149fps** |
| MobileNetV3_small | ImageNet  |224x224   |67.45    |  87.47   |**0.75ms** |**1333fps** |
| MobileNetV2_w1   | ImageNet  |224x224   |  73.08 | 91.1  |**0.58ms** |**1724fps** |

> **NOTE:** Performance measured on T4 GPU with TensorRT, using FP16 precision and batch size 1

### Pretrained Object Detection PyTorch Checkpoints

##### ** TODO - ADD HERE THE EFFICIENCY FRONTIER OBJECT-DETECTION MODELS GRAPH FOR LATENCY **


| Model | Dataset |  Resolution | mAP<sup>val<br>0.5:0.95 | Latency b1<sub>T4</sub> | Throughout b64<sub>T4</sub>  |
|--------------------- |------ | ---------- |------ | -------- |   :------: |
| YOLOv5 small | CoCo |640x640 |37.3   |**7.13ms** |**159.44fps** |
| YOLOv5 medium  | CoCo |640x640 |45.2   |**8.95ms** |**121.78fps** |

> **NOTE:** Performance measured on T4 GPU with TensorRT, using FP16 precision and batch size 1 (latency) and batch size 64 (througput)

### Pretrained Semantic Segmentation PyTorch Checkpoints

##### ** TODO - ADD HERE THE EFFICIENCY FRONTIER SEMANTIC-SEGMENTATION MODELS GRAPH FOR LATENCY **


| Model | Dataset |  Resolution | mIoU | Latency b1<sub>T4</sub> | Throughout b64<sub>T4</sub>  |
|--------------------- |------ | ---------- | ------ | -------- | :------: |
| DDRNet23   | Cityscapes |1024x2048      |78.65     |**25.48ms** |**37.4fps** |
| DDRNet23 slim   | Cityscapes |1024x2048 |76.6    |**22.24ms** |**45.7fps** |
| ShelfNet34 (with background)  | COCO Segmentation (21 classes from PASCAL) |512x512 |65.1  |**-** |**-** |

> **NOTE:** Performance measured on T4 GPU with TensorRT, using FP16 precision and  batch size 1 (latency) and batch size 64 (througput)


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

