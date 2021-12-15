# SuperGradients

## Introduction

Welcome to SuperGradients, a free open-source training library for PyTorch-based deep learning models.
There are two ways you can install it on your local machine - using this GitHub repository or using SuperGradients' private PyPi
repository.
The library lets you train models from any Computer Vision tasks or import pre-trained SOTA models, such as object detection, classification of images, and semantic segmentation for videos or images use cases.*

*Whether you are a beginner or an expert it is likely that you already have your own training script, model, loss function implementation etc.
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
   <a target="_blank" href="https://colab.research.google.com/drive/1lole-odbkD4LBnM6debK31BLP5_aILwY?usp=sharing"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />SuperGradients Quick Start in Google Colab</a>
 </td>
  <td>
   <a href="https://github.com/Deci-AI/super-gradients/blob/master/examples/quickstart.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tutorials"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>
 


## SuperGradients Walkthrough Notebook

Learn more about SuperGradients training components with our walkthrough notebook on Google Colab for an easy to use tutorial using free GPU hardware

<table class="tfo-notebook-buttons" align="left">
 <td>
   <a target="_blank" href="https://colab.research.google.com/drive/1smwh4EAgE8PwnCtwsdU8a9D9Ezfh6FQK?usp=sharing"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />SuperGradients Walkthrough in Google Colab</a>
 </td>
  <td>
   <a href="https://github.com/Deci-AI/super-gradients/blob/master/examples/SG_Walkthrough.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
 </td>
 <td>
   <a target="_blank" href="https://github.com/Deci-AI/super-gradients/tutorials"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
 </td>
</table>
 </br></br>



## Installation Methods

### Quick Installation of stable version

While being in the context of your environment, be it `venv` or `conda`, run:

```bash
  pip install git+https://github.com/Deci-AI/deci_trainer.git@stable
or
  pip install git+ssh://git@github.com/Deci-AI/super_gradients.git@stable 
```

That's it !

### Installing from GitHub

#### Prerequisites:

1. Read access to this repository. If you can read this document, you probably have it.
2. Know the version that you want to install. This can be done by viewing
   the [Releases page](https://github.com/Deci-AI/deci_trainer/releases). It's recommended to take
   the [latest one](https://github.com/Deci-AI/deci_trainer/releases/latest)

#### Installation

Let's assume the release that you would like to install is `0.0.1`. While being in the context of your environment, be
it `venv` or `conda`, run:

```bash
  pip install git+https://github.com/Deci-AI/deci_trainer.git@0.0.1
```

That's it!

Notice that the command above is using http connection. You can alternatively use SSH by running:

```bash
  pip install git+ssh://git@github.com/Deci-AI/super_gradients.git@feature/DLE-123_my_cool_feature
```

### Installing from AWS Codeartifact PyPi repository

In order to install from Codeartifact we will connect to the remote repository on AWS and modify our Pip config file.  
As we have separate repositpries for development and production, the command changes accordingly.

#### Prerequisites:

1. Make sure that you have [access to AWS](https://github.com/Deci-AI/deci-devops/blob/master/README.md#from-the-cli)
   using `aws sts get-caller-identity`.

#### 1a. Install a Release Candidate:

run:

```bash
  aws codeartifact login --tool pip --repository deci-packages --domain deci-packages --domain-owner 307629990626 --profile deci-dev
```

#### 1b. Install a Release Candidate (mutually exclusive with 1a):

run:

```bash
  aws codeartifact login --tool pip --repository deci-packages --domain deci-packages --domain-owner 487290820248 --profile deci-prod
```

#### 2. Edit `pip.conf` config file

AWS CLI configured the access token to our private PyPi repository in your ` ~/.config/pip/pip.conf` file.  
If you will open it you should see something like this:

```bash
[global]
index-url = https://aws:eyJ2ZXIiOjEsImlzdSI6MTYxNzcwMjc.....5OSwiZW5jIjoiQTcbp1rFfe_Ir_ATZUg@deci-packages-307629990626.d.codeartifact.us-east-1.amazonaws.com/pypi/deci-packages/simple/
```

We must add `extra-` prefix to `index-url` so it will become `extra-index-url = https://...`.  
You can do so by manually edit the file with your faivorite text editor, or run the coomand:
(you must have `sed` installed)

```bash

  sed -i 's/^index-url/extra-index-url/g'  ~/.config/pip/pip.conf
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



## Development Flow

### Feature and bugfix branches

When working on a branch, you will probably want to be able to test your work locally. In order to do so while not
adding noise to our PyPi repository, you can install the package directly from GitHub. There are 2 ways doing so - same
as there are for cloning - via HTTPS and via SSH.

Assuming your branch name is `feature/DLE-123_my_cool_feature` you can either:

```bash
    pip install git+https://github.com/Deci-AI/deci_trainer.git@feature/DLE-123_my_cool_feature
```

or using ssh -

```bash
     pip install git+ssh://git@github.com/Deci-AI/super_gradients.git@feature/DLE-123_my_cool_feature
```

### Debugging flow

In order to apply new changes in the code to your local machine:

1. Push the changes into the remote repository i.e. ```git push origin feature/DLE-123_my_cool_feature```
2. ``` pip uninstall deci-trainer```
3. ```pip install git+https://github.com/Deci-AI/deci_trainer.git@feature/DLE-123_my_cool_feature ```

### Merging to Master

When you are happy with your change, create a PR to master. After a code review by one of your peers, the branch will be
merged into master. That merge will trigger an automation process that will, if successful, push a release candidate
version of the package into SuperGradient's AWS Codeartifact repository. The package will be named X.Y.Zrc${CIRCLECI_BUILD}.  
In addition, the commit will be tagged with a release candidate tag - the package is ready for **staging**

### Creating a release

When we are happy with a release candidate, let's assume `0.0.1rc234`, we will checkout from that tag and create a
Release.  
The release should be named according to [SemVer2](https://semver.org/) rules. Please make sure that you understand them
before creating a release.

## Technical Debt

| Task | Jira Ticket|
| :---: | :---: |
| CI/CD does not support Patch version change | [OPS-143](https://deci.atlassian.net/browse/OPS-134) |  
| Connect Documentation like [this one](https://195-349373294-gh.circle-artifacts.com/0/docs/html/introduction/api.html) to be automatically mentioned | [OPS-135](https://deci.atlassian.net/browse/OPS-135) |  
| Add some test to make sure the CI flow is working | [OPS-136](https://deci.atlassian.net/browse/OPS-136) |  
| delete remote package if does not pass the tests | [OPS-137](https://deci.atlassian.net/browse/OPS-137) |
| Add PR numbers to RC versions in deci-trainer | [OPS-143](https://deci.atlassian.net/browse/OPS-143) |  
