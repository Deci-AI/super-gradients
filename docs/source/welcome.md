
<div>
  <img src="./assets/SG_img/SG - Horizontal.png" width="600"/>
</div>

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



To learn about making a contribution to SuperGradients, please see our [Contribution page](CONTRIBUTING.md).


