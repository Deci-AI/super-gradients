<div align="center">
  <img src="docs/assets/SG_img/SG - Horizontal.png" width="600"/>
 <br/><br/>
  
**Easily train or fine-tune SOTA computer vision models with one open source training library**
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Easily%20train%20or%20fine-tune%20SOTA%20computer%20vision%20models%20from%20one%20training%20repository&url=https://github.com/Deci-AI/super-gradients&via=deci_ai&hashtags=AI,deeplearning,computervision,training,opensource)

#### Fill our 4-question quick survey! We will raffle free SuperGradients swag between those who will participate -> [Fill Survey](https://hz8qtlvwkaw.typeform.com/to/OpKda0Qe)
______________________________________________________________________
  
  <p align="center">
  <a href="https://www.supergradients.com/">Website</a> •
  <a href="#why-use-supergradients">Why Use SG?</a> •
  <a href="https://deci-ai.github.io/super-gradients/user_guide.html#introducing-the-supergradients-library">User Guide</a> •
  <a href="https://deci-ai.github.io/super-gradients/super_gradients.common.html">Docs</a> •
  <a href="#getting-started">Getting Started Notebooks</a> •
  <a href="#transfer-learning">Transfer Learning</a> •  
  <a href="#computer-vision-models---pretrained-checkpoints">Pretrained Models</a> •
  <a href="#community">Community</a> •
  <a href="#license">License</a> •
  <a href="#deci-platform">Deci Platform</a>
</p>
<p align="center">
  <a href="https://github.com/Deci-AI/super-gradients#prerequisites"><img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue" />
  <a href="https://github.com/Deci-AI/super-gradients#prerequisites"><img src="https://img.shields.io/badge/pytorch-1.9%20%7C%201.10-blue" />
  <a href="https://pypi.org/project/super-gradients/"><img src="https://img.shields.io/pypi/v/super-gradients" />
  <a href="https://github.com/Deci-AI/super-gradients#computer-vision-models-pretrained-checkpoints" ><img src="https://img.shields.io/badge/pre--trained%20models-25-brightgreen" />
  <a href="https://github.com/Deci-AI/super-gradients/releases"><img src="https://img.shields.io/github/v/release/Deci-AI/super-gradients" />
  <a href="https://join.slack.com/t/supergradients-comm52/shared_invite/zt-10vz6o1ia-b_0W5jEPEnuHXm087K~t8Q"><img src="https://img.shields.io/badge/slack-community-blueviolet" />
  <a href="https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" />
  <a href="https://deci-ai.github.io/super-gradients/welcome.html"><img src="https://img.shields.io/badge/docs-sphinx-brightgreen" />
</p>    
</div>


# SuperGradients

## Introduction
Welcome to SuperGradients, a free, open-source training library for PyTorch-based deep learning models.
SuperGradients allows you to train or fine-tune SOTA pre-trained models for all the most commonly applied computer vision tasks with just one training library. We currently support object detection, image classification and semantic segmentation for videos and images.

Docs and full user guide[](#)
### Why use SuperGradients?
 
**Built-in SOTA Models**

Easily load and fine-tune production-ready, [pre-trained SOTA models](https://github.com/Deci-AI/super-gradients#pretrained-classification-pytorch-checkpoints) that incorporate best practices and validated hyper-parameters for achieving best-in-class accuracy.
    
**Easily Reproduce our Results**
       
Why do all the grind work, if we already did it for you? leverage tested and proven [recipes](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes) & [code examples](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/examples) for a wide range of computer vision models generated by our team of deep learning experts. Easily configure your own or use plug & play hyperparameters for training, dataset, and architecture.
    
**Production Readiness and Ease of Integration**
    
All SuperGradients models’ are production ready in the sense that they are compatible with deployment tools such as TensorRT (Nvidia) and OpenVINO (Intel) and can be easily taken into production. With a few lines of code you can easily integrate the models into your codebase.

<div align="center">
<img src="./docs/assets/SG_img/detection-demo.png" width="600px">
</div>

<div align="center">
<h3>Missing a Model or a Feature?</h3>
</div>
    
## What's New
* 【09/03/2022】 New [quick start](#quick-start-notebook---semantic-segmentation) and [transfer learning](#transfer-learning-with-sg-notebook---semantic-segmentation) example notebooks for Semantic Segmentation.
* 【07/02/2022】 We added RegSeg recipes and pre-trained models to our [Semantic Segmentation models](#pretrained-semantic-segmentation-pytorch-checkpoints).
* 【01/02/2022】 We added issue templates for feature requests and bug reporting.
* 【20/01/2022】 STDC family - new recipes added with even higher mIoU💪
* 【17/01/2022】 We have released transfer learning example [notebook](#transfer-learning-with-sg-notebook---object-detection) for object detection (YOLOv5).

Check out SG full [release notes](https://github.com/Deci-AI/super-gradients/releases).

## Comming soon
- [ ] ViT models (Vision Transformer).
- [ ] Knowledge Distillation support.
- [ ] YOLOX models (recipes, pre-trained checkpoints).
- [ ] SSD MobileNet models (recipes, pre-trained checkpoints) for edge devices deployment.
- [ ] Dali implementation.
- [ ] Integration with professional tools.

__________________________________________________________________________________________________________
### Table of Content

<!-- toc -->

- [Getting Started](#getting-started)
    - [Quick Start Notebook - Classification example](#quick-start-notebook---classification)
    - [Quick Start Notebook - Object detection example](#quick-start-notebook---object-detection)
    - [Quick Start Notebook - Semantic segmentation example](#quick-start-notebook---semantic-segmentation)
    - [Quick Start Notebook - Upload to Deci Lab example](#quick-start-notebook---model-upload-to-deci-lab)
    - [Walkthrough Notebook](#supergradients-complete-walkthrough-notebook)
- [Transfer Learning](#transfer-learning)  
    - [Transfer Learning with SG Notebook - Object detection example](#transfer-learning-with-sg-notebook---object-detection)
    - [Transfer Learning with SG Notebook - Semantic segmentation example](#transfer-learning-with-sg-notebook---semantic-segmentation)
- [Installation Methods](#installation-methods)
    - [Prerequisites](#prerequisites)
    - [Quick Installation](#quick-installation)
- [Computer Vision Models - Pretrained Checkpoints](#computer-vision-models---pretrained-checkpoints)
  - [Pretrained Classification PyTorch Checkpoints](#pretrained-classification-pytorch-checkpoints)
  - [Pretrained Object Detection PyTorch Checkpoints](#pretrained-object-detection-pytorch-checkpoints)
  - [Pretrained Semantic Segmentation PyTorch Checkpoints](#pretrained-semantic-segmentation-pytorch-checkpoints)
- [Implemented Model Architectures](#implemented-model-architectures)
- [Contributing](#contributing)
- [Citation](#citation)
- [Community](#community)
- [License](#license)
- [Deci Platform](#deci-platform)

<!-- tocstop -->
  
</details>

## Getting Started

### Start Training with Just 1 Command Line
The most simple and straightforward way to start training SOTA performance models with SuperGradients reproducible recipes. Just define your dataset path and where you want your checkpoints to be saved and you are good to go from your terminal!
    
```bash
python -m super_gradients.train_from_recipe --config-name=imagenet_regnetY architecture=regnetY800 dataset_interface.data_dir=<YOUR_Imagenet_LOCAL_PATH> ckpt_root_dir=<CHEKPOINT_DIRECTORY>
```
### Quickly Load Pre-Trained Weights for Your Desired Model with SOTA Performance
Want to try our pre-trained models on your machine? Import SuperGradients, initialize your SgModel, and load your desired architecture and pre-trained weights from our [SOTA model zoo](#computer-vision-models---pretrained-checkpoints)
    
```python
# The pretrained_weights argument will load a pre-trained architecture on the provided dataset
# This is an example of loading COCO-2017 pre-trained weights for a YOLOv5 Nano object detection model
    
import super_gradients
from super_gradients.training import SgModel

trainer = SgModel(experiment_name="yolov5n_coco_experiment",ckpt_root_dir=<CHECKPOINT_DIRECTORY>)
trainer.build_model(architecture="yolo_v5n", arch_params={"pretrained_weights": "coco", num_classes": 80})
```   
    
### Quick Start Notebook - Classification

Get started with our quick start notebook for image classification tasks on Google Colab for a quick and easy start using free GPU hardware.

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://bit.ly/3ufnsgT"><img src="./docs/assets/SG_img/colab_logo.png" />Classification Quick Start in Google Colab</a>
 </td>
  <td>
   <a href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/SG_quickstart_classification.ipynb"><img src="./docs/assets/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/examples"><img src="./docs/assets/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>

### Quick Start Notebook - Object Detection

Get started with our quick start notebook for object detection tasks on Google Colab for a quick and easy start using free GPU hardware.

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://bit.ly/3wqMsEM"><img src="./docs/assets/SG_img/colab_logo.png" />Detection Quick Start in Google Colab</a>
 </td>
  <td>
   <a href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/SG_quickstart_detection.ipynb"><img src="./docs/assets/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/examples"><img src="./docs/assets/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>

### Quick Start Notebook - Semantic Segmentation

Get started with our quick start notebook for semantic segmentation tasks on Google Colab for a quick and easy start using free GPU hardware.

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://bit.ly/3Jp7w1U"><img src="./docs/assets/SG_img/colab_logo.png" />Segmentation Quick Start in Google Colab</a>
 </td>
  <td>
   <a href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/SG_quickstart_segmentation.ipynb"><img src="./docs/assets/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/examples"><img src="./docs/assets/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>

### Quick Start Notebook - Model Upload to Deci Lab

Get Started with an example of how to upload to Deci Lab a freshly trained model
<table class="tfo-notebook-buttons" align="left">
  <tbody>
    <tr>
      <td vertical-align="middle">
        <img src="./docs/assets/SG_img/colab_logo.png" />
        <a target="_blank" href="https://colab.research.google.com/drive/1cNvakn8ttLhD9g8IMbe51PBvk-vJ40Oi?usp=sharing&utm_campaign=SG%20github%20repo&utm_source=Google%20Colab&utm_medium=GitHub%20Repo&utm_content=Quickstart%20trainig%20with20model%20upload%20notebook%20-%20README.md">
          Classification Quick Start in Google Colab
        </a>
      </td>
      <td vertical-align="middle">
        <img src="./docs/assets/SG_img/download_logo.png" />
        <a href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/SG_quickstart_model_upload_deci_lab.ipynb">
          Download notebook
        </a>
      </td>
      <td>
        <img src="./docs/assets/SG_img/GitHub_logo.png" />
        <a target="_blank" href="https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/deci_lab_export_example/deci_lab_export_example.py">
          View source on GitHub
        </a>
      </td>
    </tr>
  </tbody>
</table>
<br>
 
### SuperGradients Complete Walkthrough Notebook

Learn more about SuperGradients training components with our walkthrough notebook on Google Colab for an easy to use tutorial using free GPU hardware

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://bit.ly/3JspSPF"><img src="./docs/assets/SG_img/colab_logo.png" />SuperGradients Walkthrough in Google Colab</a>
 </td>
  <td>
   <a href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/SG_Walkthrough.ipynb"><img src="./docs/assets/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/examples"><img src="./docs/assets/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>
 
 
 ## Transfer Learning
 ### Transfer Learning with SG Notebook - Object Detection

Learn more about SuperGradients transfer learning or fine tuning abilities with our COCO pre-trained YoloV5nano fine tuning into a sub-dataset of PASCAL VOC example notebook on Google Colab for an easy to use tutorial using free GPU hardware

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://bit.ly/3iGvnP7"><img src="./docs/assets/SG_img/colab_logo.png" />Detection Transfer Learning in Google Colab</a>
 </td>
  <td>
   <a href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/SG_transfer_learning_object_detection.ipynb"><img src="./docs/assets/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/examples"><img src="./docs/assets/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>
 
### Transfer Learning with SG Notebook - Semantic Segmentation
Learn more about SuperGradients transfer learning or fine tuning abilities with our Citiscapes pre-trained RegSeg48 fine tuning into a sub-dataset of Supervisely example notebook on Google Colab for an easy to use tutorial using free GPU hardware

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://bit.ly/37P04PN"><img src="./docs/assets/SG_img/colab_logo.png" />Segmentation Transfer Learning in Google Colab</a>
 </td>
  <td>
   <a href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/SG_transfer_learning_semantic_segmentation.ipynb"><img src="./docs/assets/SG_img/download_logo.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/examples"><img src="./docs/assets/SG_img/GitHub_logo.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>

## Installation Methods
### Prerequisites
<details>
  
<summary>General requirements</summary>
  
- Python 3.7, 3.8 or 3.9 installed.
- torch>=1.9.0
  - https://pytorch.org/get-started/locally/
- The python packages that are specified in requirements.txt;

</details>
    
<details>
  
<summary>To train on nvidia GPUs</summary>
  
- [Nvidia CUDA Toolkit >= 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu)
- CuDNN >= 8.1.x
- Nvidia Driver with CUDA >= 11.2 support (≥460.x)
  
</details>
    
### Quick Installation

<details>
  
<summary>Install stable version using PyPi</summary>

See in [PyPi](https://pypi.org/project/super-gradients/)
```bash
pip install super-gradients
```

That's it !

</details>
    
<details>
  
<summary>Install using GitHub</summary>


```bash
pip install git+https://github.com/Deci-AI/super-gradients.git@stable
```

</details> 


## Computer Vision Models - Pretrained Checkpoints

### Pretrained Classification PyTorch Checkpoints


| Model | Dataset |  Resolution |    Top-1    |    Top-5   | Latency (HW)*<sub>T4</sub>  | Latency (Production)**<sub>T4</sub> |Latency (HW)*<sub>Jetson Xavier NX</sub>  | Latency (Production)**<sub>Jetson Xavier NX</sub> | Latency <sub>Cascade Lake</sub>  |
|------------ | ------ | ---------- |----------- | ----------- | ----------- |---------- |----------- | ----------- | :------: |
| EfficientNet B0 | ImageNet | 224x224 |  77.62  | 93.49 |**0.93ms** |**1.38ms** | **-** * |**-**|**3.44ms** |
| RegNet Y200 | ImageNet  |224x224 |  70.88   | 89.35 |**0.63ms** | **1.08ms** | **2.16ms** |**2.47ms**|**2.06ms** |
| RegNet Y400  | ImageNet |224x224 |  74.74   | 91.46 |**0.80ms** | **1.25ms** |**2.62ms** |**2.91ms** |**2.87ms** |
| RegNet Y600  | ImageNet |224x224 |  76.18   | 92.34 |**0.77ms** | **1.22ms** |**2.64ms** |**2.93ms** |**2.39ms** |
| RegNet Y800  | ImageNet |224x224 |  77.07  |  93.26 |**0.74ms** | **1.19ms** |**2.77ms** |**3.04ms** |**2.81ms** |
| ResNet 18   | ImageNet  |224x224   |  70.6   |   89.64 |**0.52ms** | **0.95ms** |**2.01ms**|**2.30ms** |**4.56ms** |
| ResNet 34  | ImageNet  |224x224   |  74.13   |   91.7  |**0.92ms**  |**1.34ms** |**3.57ms**|**3.87ms** | **7.64ms** |
| ResNet 50  | ImageNet  |224x224   |  79.47  |   93.0  |**1.03ms** | **1.44ms** | **4.78ms**|**5.10ms** |**9.25ms** |
| MobileNet V3_large-150 epochs | ImageNet  |224x224   |  73.79    |   91.54  |**0.67ms** | **1.11ms** |**2.42ms** |**2.71ms** |**1.76ms** |
| MobileNet V3_large-300 epochs  | ImageNet  |224x224   |  74.52    |  91.92 |**0.67ms** | **1.11ms** |**2.42ms** |**2.71ms** |**1.76ms** |
| MobileNet V3_small | ImageNet  |224x224   |67.45    |  87.47   |**0.55ms** | **0.96ms** |**2.01ms** *|**2.35ms** |**1.06ms** |
| MobileNet V2_w1   | ImageNet  |224x224   |  73.08 | 91.1  |**0.46 ms**| **0.89ms** |**1.65ms** *|**1.90ms** | **1.56ms** |
> **NOTE:** <br/>
> - Latency (HW)* - Hardware performance (not including IO)<br/>
> - Latency (Production)** - Production Performance (including IO)
> - Performance measured for T4 and Jetson Xavier NX with TensorRT, using FP16 precision and batch size 1
> - Performance measured for Cascade Lake CPU with OpenVINO, using FP16 precision and batch size 1



### Pretrained Object Detection PyTorch Checkpoints


| Model | Dataset |  Resolution | mAP<sup>val<br>0.5:0.95 | Latency (HW)*<sub>T4</sub>  | Latency (Production)**<sub>T4</sub> |Latency (HW)*<sub>Jetson Xavier NX</sub>  | Latency (Production)**<sub>Jetson Xavier NX</sub> | Latency <sub>Cascade Lake</sub>  |
|------------- |------ | ---------- |------ | -------- |------ | ---------- |------ | :------: |
| YOLOv5 nano | COCO |640x640 |27.7  |**1.48ms** |**5.43ms**|**9.28ms** |**17.44ms** |**21.71ms**|
| YOLOv5 small | COCO |640x640 |37.3 |**2.29ms** |**6.14ms**|**14.31ms** |**22.50ms** |**34.10ms**|
| YOLOv5 medium| COCO |640x640 |45.2 |**4.60ms** |**8.10ms**|**26.76ms** |**34.95ms** |**65.86ms**|
| YOLOv5 large | COCO |640x640 |48.0 |**7.20ms** |**10.28ms**|**43.89ms** |**51.92ms** |**122.97ms**|
  

> **NOTE:** <br/>
> - Latency (HW)* - Hardware performance (not including IO)<br/>
> - Latency (Production)** - Production Performance (including IO)
> - Latency performance measured for T4 and Jetson Xavier NX with TensorRT, using FP16 precision and batch size 1
> - Latency performance measured for Cascade Lake CPU with OpenVINO, using FP16 precision and batch size 1

### Pretrained Semantic Segmentation PyTorch Checkpoints


| Model | Dataset |  Resolution | mIoU | Latency b1<sub>T4</sub> | Latency b1<sub>T4</sub> including IO |
|--------------------- |------ | ---------- | ------ | -------- | :------: |
| DDRNet 23   | Cityscapes |1024x2048   |78.65  |**7.62ms** |**25.94ms**|
| DDRNet 23 slim   | Cityscapes |1024x2048 |76.6  |**3.56ms** |**22.80ms**|
| STDC 1-Seg50   | Cityscapes | 512x1024 |74.36 |**2.83ms** |**12.57ms**|
| STDC 1-Seg75   | Cityscapes | 768x1536 |76.87  |**5.71ms** |**26.70ms**|
| STDC 2-Seg50   | Cityscapes | 512x1024 |75.27 |**3.74ms** |**13.89ms**
| STDC 2-Seg75   | Cityscapes | 768x1536 |78.93 |**7.35ms** |**28.18ms**|
| RegSeg (exp48)   | Cityscapes | 1024x2048 |78.15 |**13.09ms** |**41.88ms**|
| Larger RegSeg (exp53)   | Cityscapes | 1024x2048 |79.2|**24.82ms** |**51.87ms**|
| ShelfNet LW 34 | COCO Segmentation (21 classes from PASCAL including background) |512x512 |65.1  |**-** |**-** |


> **NOTE:** Performance measured on T4 GPU with TensorRT, using FP16 precision and batch size 1 (latency), and not including IO

## Implemented Model Architectures 
  
### Image Classification
  
- [DensNet (Densely Connected Convolutional Networks)](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/densenet.py) - Densely Connected Convolutional Networks [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)
- [DPN](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/dpn.py) - Dual Path Networks [https://arxiv.org/pdf/1707.01629](https://arxiv.org/pdf/1707.01629)
- [EfficientNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/efficientnet.py) - [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- [GoogleNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/googlenet.py) - [https://arxiv.org/pdf/1409.4842](https://arxiv.org/pdf/1409.4842)
- [LeNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/lenet.py) - [https://yann.lecun.com/exdb/lenet/](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [MobileNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/mobilenet.py) - Efficient Convolutional Neural Networks for Mobile Vision Applications [https://arxiv.org/pdf/1704.04861](https://arxiv.org/pdf/1704.04861)
- [MobileNet v2](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/mobilenetv2.py) - [https://arxiv.org/pdf/1801.04381](https://arxiv.org/pdf/1801.04381) 
- [MobileNet v3](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/mobilenetv3.py) - [https://arxiv.org/pdf/1905.02244](https://arxiv.org/pdf/1905.02244)
- [PNASNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/pnasnet.py) - Progressive Neural Architecture Search Networks [https://arxiv.org/pdf/1712.00559](https://arxiv.org/pdf/1712.00559)
- [Pre-activation ResNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/preact_resnet.py) - [https://arxiv.org/pdf/1603.05027](https://arxiv.org/pdf/1603.05027)  
- [RegNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/regnet.py) - [https://arxiv.org/pdf/2003.13678.pdf](https://arxiv.org/pdf/2003.13678.pdf) 
- [RepVGG](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/repvgg.py) - Making VGG-style ConvNets Great Again [https://arxiv.org/pdf/2101.03697.pdf](https://arxiv.org/pdf/2101.03697.pdf) 
- [ResNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/resnet.py) - Deep Residual Learning for Image Recognition [https://arxiv.org/pdf/1512.03385](https://arxiv.org/pdf/1512.03385)  
- [ResNeXt](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/resnext.py) - Aggregated Residual Transformations for Deep Neural Networks [https://arxiv.org/pdf/1611.05431](https://arxiv.org/pdf/1611.05431)
- [SENet ](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/senet.py) - Squeeze-and-Excitation Networks[https://arxiv.org/pdf/1709.01507](https://arxiv.org/pdf/1709.01507)
- [ShuffleNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/shufflenet.py) - [https://arxiv.org/pdf/1707.01083](https://arxiv.org/pdf/1707.01083)
- [ShuffleNet v2](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/shufflenetv2.py) - Efficient Convolutional Neural Network for Mobile
Devices[https://arxiv.org/pdf/1807.11164](https://arxiv.org/pdf/1807.11164)
- [VGG](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/classification_models/vgg.py) - Very Deep Convolutional Networks for Large-scale Image Recognition [https://arxiv.org/pdf/1409.1556](https://arxiv.org/pdf/1409.1556)
  
  
### Object Detection
  
- [CSP DarkNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/csp_darknet53.py)
- [DarkNet-53](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/darknet53.py)
- [SSD (Single Shot Detector)](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/ssd.py) - [https://arxiv.org/pdf/1512.02325](https://arxiv.org/pdf/1512.02325)
- [YOLO v3](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/yolov3.py) - [https://arxiv.org/pdf/1804.02767](https://arxiv.org/pdf/1804.02767)
- [YOLO v5](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/yolov5.py) - [by Ultralytics](https://docs.ultralytics.com/)
  
  
### Semantic Segmentation 
  
- [DDRNet (Deep Dual-resolution Networks)](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py) - [https://arxiv.org/pdf/2101.06085.pdf](https://arxiv.org/pdf/2101.06085.pdf)
- [LadderNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/laddernet.py) - Multi-path networks based on U-Net for medical image segmentation [https://arxiv.org/pdf/1810.07810](https://arxiv.org/pdf/1810.07810)
- [RegSeg](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/regseg.py) - Rethink Dilated Convolution for Real-time Semantic Segmentation [https://arxiv.org/pdf/2111.09957](https://arxiv.org/pdf/2111.09957)
- [ShelfNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/shelfnet.py) - [https://arxiv.org/pdf/1811.11254](https://arxiv.org/pdf/1811.11254)
- [STDC](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/stdc.py) - Rethinking BiSeNet For Real-time Semantic Segmentation [https://arxiv.org/pdf/2104.13188](https://arxiv.org/pdf/2104.13188)
  
</details>
  
## Documentation

Check SuperGradients [Docs](https://deci-ai.github.io/super-gradients/welcome.html) for full documentation, user guide, and examples.
  
## Contributing

To learn about making a contribution to SuperGradients, please see our [Contribution page](CONTRIBUTING.md).

Our awesome contributors:
    
<a href="https://github.com/Deci-AI/super-gradients/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Deci-AI/super-gradients" />
</a>


<br/>Made with [contrib.rocks](https://contrib.rocks).

## Citation

If you are using SuperGradients library or benchmarks in your research, please cite SuperGradients deep learning training library.

## Community

If you want to be a part of SuperGradients growing community, hear about all the exciting news and updates, need help, request for advanced features,
    or want to file a bug or issue report, we would love to welcome you aboard!

* Slack is the place to be and ask questions about SuperGradients and get support. [Click here to join our Slack](
  https://join.slack.com/t/supergradients-comm52/shared_invite/zt-10vz6o1ia-b_0W5jEPEnuHXm087K~t8Q)
    
* To report a bug, [file an issue](https://github.com/Deci-AI/super-gradients/issues) on GitHub.

* Join the [SG Newsletter](https://www.supergradients.com/#Newsletter)
  for staying up to date with new features and models, important announcements, and upcoming events.
    
* For a short meeting with us, use this [link](https://calendly.com/ofer-baratz-deci/15min) and choose your preferred time.

## License

This project is released under the [Apache 2.0 license](LICENSE).
    

    
__________________________________________________________________________________________________________


## Deci Platform

Deci Platform is our end to end platform for building, optimizing and deploying deep learning models to production.

Sign up for our [FREE Community Tier](https://console.deci.ai/) to enjoy immediate improvement in throughput, latency, memory footprint and model size.

Features:
- Automatically compile and quantize your models with just a few clicks (TensorRT, OpenVINO).
- Gain up to 10X improvement in throughput, latency, memory and model size. 
- Easily benchmark your models’ performance on different hardware and batch sizes.
- Invite co-workers to collaborate on models and communicate your progress.
- Deci supports all common frameworks and Hardware, from Intel CPUs to Nvidia's GPUs and Jetsons.

Sign up for Deci Platform for free [here](https://console.deci.ai/) 

