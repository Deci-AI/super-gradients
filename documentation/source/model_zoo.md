# Model Zoo

## Computer Vision Models - Pretrained Checkpoints

You can load any of our pretrained model in 2 lines of code:
```python
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLOX_S, pretrained_weights="coco")
```

All the available models are listed in the column `Model name`.


### Pretrained Classification PyTorch Checkpoints


| Model                         | Model name            | Dataset     | Resolution | Top-1  | Top-5   | Latency (HW)*<sub>T4</sub> | Latency (Production)**<sub>T4</sub> | Latency (HW)*<sub>Jetson Xavier NX</sub> | Latency (Production)**<sub>Jetson Xavier NX</sub> | Latency <sub>Cascade Lake</sub> |
|-------------------------------|-----------------------|-------------|------------|--------|---------|----------------------------|-------------------------------------|------------------------------------------|---------------------------------------------------|:-------------------------------:|
| ViT base                      | vit_base              | ImageNet21K | 224x224    | 84.15  | -       | **4.46ms**                 | **4.60ms**                          | **-** *                                  | **-**                                             |          **57.22ms**            |
| ViT large                     | vit_large             | ImageNet21K | 224x224    | 85.64  | -       | **12.81ms**                | **13.19ms**                         | **-** *                                  | **-**                                             |          **187.22ms**           |
| BEiT                          | beit_base_patch16_224 | ImageNet21K | 224x224    | -      | -       | **-ms**                    | **-ms**                             | **-** *                                  | **-**                                             |             **-ms**             |
| EfficientNet B0               | efficientnet_b0       | ImageNet    | 224x224    | 77.62  | 93.49   | **0.93ms**                 | **1.38ms**                          | **-** *                                  | **-**                                             |           **3.44ms**            |
| RegNet Y200                   | regnetY200            | ImageNet    | 224x224    | 70.88  | 89.35   | **0.63ms**                 | **1.08ms**                          | **2.16ms**                               | **2.47ms**                                        |           **2.06ms**            |
| RegNet Y400                   | regnetY400            | ImageNet    | 224x224    | 74.74  | 91.46   | **0.80ms**                 | **1.25ms**                          | **2.62ms**                               | **2.91ms**                                        |           **2.87ms**            |
| RegNet Y600                   | regnetY600            | ImageNet    | 224x224    | 76.18  | 92.34   | **0.77ms**                 | **1.22ms**                          | **2.64ms**                               | **2.93ms**                                        |           **2.39ms**            |
| RegNet Y800                   | regnetY800            | ImageNet    | 224x224    | 77.07  | 93.26   | **0.74ms**                 | **1.19ms**                          | **2.77ms**                               | **3.04ms**                                        |           **2.81ms**            |
| ResNet 18                     | resnet18              | ImageNet    | 224x224    | 70.6   | 89.64   | **0.52ms**                 | **0.95ms**                          | **2.01ms**                               | **2.30ms**                                        |           **4.56ms**            |
| ResNet 34                     | resnet34              | ImageNet    | 224x224    | 74.13  | 91.7    | **0.92ms**                 | **1.34ms**                          | **3.57ms**                               | **3.87ms**                                        |           **7.64ms**            |
| ResNet 50                     | resnet50              | ImageNet    | 224x224    | 81.91  | 93.0    | **1.03ms**                 | **1.44ms**                          | **4.78ms**                               | **5.10ms**                                        |           **9.25ms**            |
| MobileNet V3_large-300 epochs | mobilenet_v3_large    | ImageNet    | 224x224    | 74.52  | 91.92   | **0.67ms**                 | **1.11ms**                          | **2.42ms**                               | **2.71ms**                                        |           **1.76ms**            |
| MobileNet V3_small            | mobilenet_v3_small    | ImageNet    | 224x224    | 67.45  | 87.47   | **0.55ms**                 | **0.96ms**                          | **2.01ms** *                             | **2.35ms**                                        |           **1.06ms**            |
| MobileNet V2_w1               | mobilenet_v2          | ImageNet    | 224x224    | 73.08  | 91.1    | **0.46 ms**                | **0.89ms**                          | **1.65ms** *                             | **1.90ms**                                        |           **1.56ms**            |

> **NOTE:** <br/>
> - Latency (HW)* - Hardware performance (not including IO)<br/>
> - Latency (Production)** - Production Performance (including IO)
> - Performance measured for T4 and Jetson Xavier NX with TensorRT, using FP16 precision and batch size 1
> - Performance measured for Cascade Lake CPU with OpenVINO, using FP16 precision and batch size 1



### Pretrained Object Detection PyTorch Checkpoints


| Model                 | Model Name            | Dataset | Resolution | mAP<sup>val<br>0.5:0.95  | Latency (HW)*<sub>T4</sub>    | Latency (Production)**<sub>T4</sub> | Latency (HW)*<sub>Jetson Xavier NX</sub> | Latency (Production)**<sub>Jetson Xavier NX</sub> | Latency <sub>Cascade Lake</sub> |
|-----------------------|-----------------------|---------|------------|--------------------------|-------------------------------|-------------------------------------|------------------------------------------|---------------------------------------------------|:-------------------------------:|
| YOLO-NAS S            | yolo_nas_s            | COCO    | 640x640    | 47.5(FP16) 47.03(INT8)   | **3.21(FP16)** **2.36(INT8)** |
| YOLO-NAS M            | yolo_nas_m            | COCO    | 640x640    | 51.55(FP16) 51.0(INT8)   | **5.85(FP16)** **3.78(INT8)** |
| YOLO-NAS L            | yolo_nas_l            | COCO    | 640x640    | 52.22(FP16) 52.1(INT8)   | **7.87(FP16)** **4.78(INT8)** |
| PP-YOLOE small        | ppyoloe_s             | COCO    | 640x640    | 42.52                    | **2.39ms**                    | **4.3ms**                           | **14.28ms**                              | **14.99ms**                                       |              **-**              |
| PP-YOLOE medium       | ppyoloe_m             | COCO    | 640x640    | 47.11                    | **5.16ms**                    | **7.05ms**                          | **32.71ms**                              | **33.46ms**                                       |              **-**              |
| PP-YOLOE large        | ppyoloe_l             | COCO    | 640x640    | 49.48                    | **7.65ms**                    | **9.59ms**                          | **51.13ms**                              | **50.39ms**                                       |              **-**              |
| PP-YOLOE x-large      | ppyoloe_x             | COCO    | 640x640    | 51.15                    | **14.04ms**                   | **15.96ms**                         | **94.92ms**                              | **94.22ms**                                       |              **-**              |
| YOLOX nano            | yolox_n               | COCO    | 640x640    | 26.77                    | **2.47ms**                    | **4.09ms**                          | **11.49ms**                              | **12.97ms**                                       |              **-**              |
| YOLOX tiny            | yolox_t               | COCO    | 640x640    | 37.18                    | **3.16ms**                    | **4.61ms**                          | **15.23ms**                              | **19.24ms**                                       |              **-**              |
| YOLOX small           | yolox_s               | COCO    | 640x640    | 40.47                    | **3.58ms**                    | **4.94ms**                          | **18.88ms**                              | **22.48ms**                                       |              **-**              |
| YOLOX medium          | yolox_m               | COCO    | 640x640    | 46.4                     | **6.40ms**                    | **7.65ms**                          | **39.22ms**                              | **44.5ms**                                        |              **-**              |
| YOLOX large           | yolox_l               | COCO    | 640x640    | 49.25                    | **10.07ms**                   | **11.12ms**                         | **68.73ms**                              | **77.01ms**                                       |              **-**              |
| SSD lite MobileNet v2 | ssd_lite_mobilenet_v2 | COCO    | 320x320    | 21.5                     | **0.77ms**                    | **1.40ms**                          | **5.28ms**                               | **6.44ms**                                        |            **4.13ms**           |
| SSD lite MobileNet v1 | ssd_mobilenet_v1      | COCO    | 320x320    | 24.3                     | **1.55ms**                    | **2.84ms**                          | **8.07ms**                               | **9.14ms**                                        |           **22.76ms**           |

> **NOTE:** <br/>
> - Latency (HW)* - Hardware performance (not including IO)<br/>
> - Latency (Production)** - Production Performance (including IO)
> - Latency performance measured for T4 and Jetson Xavier NX with TensorRT, using FP16 precision and batch size 1
> - Latency performance measured for Cascade Lake CPU with OpenVINO, using FP16 precision and batch size 1

### Pretrained Semantic Segmentation PyTorch Checkpoints

| Model                 | Model Name        | Dataset    | Resolution | mIoU  | Latency b1<sub>T4</sub> | Latency b1<sub>T4</sub> including IO | Latency (Production)**<sub>Jetson Xavier NX</sub> | 
|-----------------------|-------------------|------------|------------|-------|-------------------------|--------------------------------------|:-------------------------------------------------:|
| PP-LiteSeg B50        | pp_lite_b_seg50   | Cityscapes | 512x1024   | 76.48 | **4.18ms**              | **31.22ms**                          |                    **31.69ms**                    |
| PP-LiteSeg B75        | pp_lite_b_seg75   | Cityscapes | 768x1536   | 78.52 | **6.84ms**              | **33.69ms**                          |                    **49.89ms**                    |
| PP-LiteSeg T50        | pp_lite_t_seg50   | Cityscapes | 512x1024   | 74.92 | **3.26ms**              | **30.33ms**                          |                    **26.20ms**                    |
| PP-LiteSeg T75        | pp_lite_t_seg75   | Cityscapes | 768x1536   | 77.56 | **5.20ms**              | **32.28ms**                          |                    **38.03ms**                    |
| DDRNet 23 slim        | ddrnet_23_slim    | Cityscapes | 1024x2048  | 79.41 | **5.74ms**              | **32.01ms**                          |                    **45.18ms**                    |
| DDRNet 23             | ddrnet_23         | Cityscapes | 1024x2048  | 81.48 | **12.74ms**             | **39.01ms**                          |                   **106.26ms**                    |
| DDRNet 39             | ddrnet_39         | Cityscapes | 1024x2048  | 81.32 | **23.57ms**             | **52.41ms**                          |                   **145.79ms**                    |
| STDC 1-Seg50          | stdc1_seg50       | Cityscapes | 512x1024   | 75.11 | **3.34ms**              | **30.12ms**                          |                    **27.54ms**                    |
| STDC 1-Seg75          | stdc1_seg75       | Cityscapes | 768x1536   | 77.8  | **5.53ms**              | **32.490ms**                         |                     **43.88**                     |
| STDC 2-Seg50          | stdc2_seg50       | Cityscapes | 512x1024   | 76.44 | **4.12ms**              | **30.94ms**                          |                    **32.03ms**                    |
| STDC 2-Seg75          | stdc2_seg75       | Cityscapes | 768x1536   | 78.93 | **6.95ms**              | **33.89ms**                          |                   **54.48ms**                     |
| RegSeg (exp48)        | regseg48          | Cityscapes | 1024x2048  | 78.15 | **12.03ms**             | **38.91ms**                          |                    **78.20ms**                    |

> **NOTE:** <br/>
> - Performance measured on T4 GPU with TensorRT, using FP16 precision and batch size 1 (latency), and not including IO
> - For resolutions below 1024x2048 we first resize the input to the inference resolution and then resize the predictions to 1024x2048. The time of resizing is included in the measurements so that the practical input-size is 1024x2048.
> - DDRNet23 and DDRNet23_Slim results were achieved with channel wise knowledge distillation training recipe.


### Pretrained Pose Estimation PyTorch Checkpoints

| Model           | Model Name      | Dataset     | Resolution | AP (No TTA / H-Flip TTA / H-Flip TTA+Rescoring) | Latency b1<sub>T4</sub> | Latency b1<sub>T4</sub> including IO | Latency (Production)**<sub>Jetson Xavier NX</sub> | 
|-----------------|-----------------|-------------|------------|-------------------------------------------------|-------------------------|--------------------------------------|:-------------------------------------------------:|
| DEKR_W32_NO_DC  | dekr_w32_no_dc  | COCO2017 PE | 640x640    | 63.08 / 64.96 / 67.32                           | 13.29 ms                | 15.31 ms                             | 75.99 ms                                          |


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
- [YOLOX](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/yolox.py) - [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)
- [PP-YoloE](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/pp_yolo_e/pp_yolo_e.py) - [https://arxiv.org/abs/2203.16250](https://arxiv.org/abs/2203.16250)

### Semantic Segmentation 

- [PP-LiteSeg](https://bit.ly/3RrtMMO) - [https://arxiv.org/pdf/2204.02681v1.pdf](https://arxiv.org/pdf/2204.02681v1.pdf) 
- [DDRNet (Deep Dual-resolution Networks)](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py) - [https://arxiv.org/pdf/2101.06085.pdf](https://arxiv.org/pdf/2101.06085.pdf)
- [LadderNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/laddernet.py) - Multi-path networks based on U-Net for medical image segmentation [https://arxiv.org/pdf/1810.07810](https://arxiv.org/pdf/1810.07810)
- [RegSeg](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/regseg.py) - Rethink Dilated Convolution for Real-time Semantic Segmentation [https://arxiv.org/pdf/2111.09957](https://arxiv.org/pdf/2111.09957)
- [ShelfNet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/shelfnet.py) - [https://arxiv.org/pdf/1811.11254](https://arxiv.org/pdf/1811.11254)
- [STDC](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/stdc.py) - Rethinking BiSeNet For Real-time Semantic Segmentation [https://arxiv.org/pdf/2104.13188](https://arxiv.org/pdf/2104.13188)
  

### Pose Estimation
- [HRNet DEKR](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) - Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression [https://arxiv.org/pdf/2104.02300.pdf](https://arxiv.org/pdf/2104.02300.pdf)
