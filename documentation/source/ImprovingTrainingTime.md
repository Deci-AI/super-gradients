# Improving Training Time

## Mixed Precision Training

Automatic mixed precision (AMP) is a feature in PyTorch that enables the use of lower-precision data types, such as float16, in deep learning models for improved memory and computation efficiency. 
It automatically casts the model's parameters and buffers to a lower-precision data type, and dynamically rescales the activations to prevent underflow or overflow. 

Most modern GPUs [support](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix) float16 operations natively, and can therefore accelerate the training process.

To use `AMP` in SuperGradients, you simply need to set `mixed_precision=True` in your training_hyperparams.

**In python script**
```python
from super_gradients import Trainer

trainer = Trainer("experiment_name")
model = ...

training_hyperparams = {"mixed_precision": True, ...:...}

trainer.train(model=model, training_hyperparams=training_hyperparams, ...)
```

**In recipe**
```yaml
# my_training_hyperparams.yaml

mixed_precision: True # Whether to use mixed precision or not.
```


## Torch Compile

PyTorch 2.0 introduced new [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) API which can be used to improve the training time of the model. 
This API can be used to fuse the operations in the model graph and optimize the model for the target device.

SuperGradients support the `torch.compile` API and can be used to improve the training time of the model. 
Here we report the relative improvement (reduction) of training time for several models and tasks. We measure the training time of one epoch. 
This includes iteration over training and validation datasets, loss & metric computation. Essentially all steps that are performed during training.
Please note that the improvement vary depending on the model architecture, dataset, and training hyperparameters.

| Task                  | Recipe                          | Baseline (1 GPU) | Baseline (8 GPU) | 1 GPU With Compile | 8 GPU With Compile | Improvement, % (1 GPU) | Improvement, % (8 GPU) |
|-----------------------|---------------------------------|------------------|------------------|--------------------|--------------------|------------------------|------------------------|
| Semantic Segmentation | cityscapes_pplite_seg75         | 270.63           | 49.95            | 119.11             | 35.91              | 56%                    | 18%                    |
| Semantic Segmentation | cityscapes_regseg48             | 125.14           | 44.959           | 108.57             | 44.55              | 13.2%                  | 0.9%                   |
| Semantic Segmentation | cityscapes_segformer            | 199.97           | 46.21            | 162.52             | 43.71              | 18.7%                  | 5.4%                   |
| Semantic Segmentation | cityscapes_stdc_seg75           | 425.19           | 73.07            | 153.16             | 45.89              | 63.9%                  | 37.19%                 |
| Semantic Segmentation | cityscapes_ddrnet               | 226.51           | 51.78            | 174.11             | 48.29              | 23.1%                  | 7.3%                   |
|                       |                                 |                  |                  |                    |                    |                        |                        |
| Object Detection      | coco2017_yolo_nas_s             | 1509             | 384.41           | 1379               | 376.10             | 8.6%                   | 2.42%                  |
| Object Detection      | coco2017_yolo_nas_m             | 2363             | 537.24           | 2090               | 508.40             | 11.5%                  | 0.19%                  |
| Object Detection      | coco2017_yolo_nas_l             | 3193             | 764.17           | 2869               | 745.58             | 10.14%                 | 2.43%                  |
| Object Detection      | coco2017_ppyoloe_s/m/l/x        | N/A              | N/A              |                    | N/A                | N/A                    | N/A                    |
| Object Detection      | coco2017_yolox_n/t/s/m/l/x      | N/A              | N/A              |                    | N/A                | N/A                    | N/A                    |
| Object Detection      | coco2017_ssd_lite_mobilenet_v2  | N/A              | N/A              |                    | N/A                | N/A                    | N/A                    |
|                       |                                 |                  |                  |                    |                    |                        |                        |
| Classification        | imagenet_efficientnet           |                  | 425.61           |                    | 408.39             |                        | 4.1%                   |
| Classification        | imagenet_mobilenetv3_large      |                  | 373.73           |                    | 374.51             |                        | -0.2%                  |
| Classification        | imagenet_regnetY                |                  | 406.86           |                    | 383.04             |                        | 5.8%                   |
| Classification        | imagenet_repvgg                 |                  | 407.19           |                    | 387.00             |                        | 4.9%                   |
| Classification        | imagenet_resnet50               |                  | 481.36           |                    | 480.29             |                        | 0.22%                  |
| Classification        | imagenet_vit_base               | N/A              | N/A              | N/A                | N/A                | N/A                    | N/A                    |
| Classification        | imagenet_vit_large              | N/A              | N/A              | N/A                | N/A                | N/A                    | N/A                    |

In the table above, number are reported as speedup compared to the baseline training time. 
Both experiments were run on 8x 3090 GPUs using PyTorch 2.0 with CUDA 11.8. 
Training was done for 5 epochs and median value was picked to compute the speedup. 
All experiments conducted with mixed precision (AMP) enabled, and SyncBN and EMA disabled.
Improvement percentage computed as follows: `100 * (baseline_time - compile_time) / baseline_time`.

To leverage use of compiled models in SuperGradients one need to pass the `torch_compile: True` option to training hyperparameters:

```bash
python -m super_gradients.train_from_recipe --config-name=... training_hyperparams.torch_compile=True
```

In the YAML recipe:

```yaml
# my_training_recipe.yaml
training_hyperparams:
  torch_compile: True 
  torch_compile_mode: default | reduce-overhead | max-autotune
```


Or programmatically:

```python
from super_gradients.training import Trainer

trainer = Trainer(
    ...,
    training_hyperparams = {
        "torch_compile": True,
        ...
        }
    )
```



### Avoiding common pitfalls:

* Don't use EMA with `torch.compile`.
* Don't use SyncBN with `torch.compile`.
* You may need to reduce batch size during training by quite a lot (Up to 2x)
* Training with mixed precision gives the best performance boost.

#### Exponential moving average `EMA` and Torch Compile

Torch Compile is still in its early stages and has some limitations. Not every model can be compiled. 
Additionally, some training features can conflict with `torch.compile`. 
Here is what we found so far:

* Exponential moving average `EMA` is incompatible with `torch.compile` (At the moment of writing, this is true for SG release 3.1.2). 
  If you want to use `torch.compile` in your training, you need to disable `EMA` when using `torch.compile`.
  ```yaml
  training_hyperparams:
    torch_compile: True 
    ema: False
  ```
  
#### Sync BatchNorm and Torch Compile 

In short: 

When training using DDP, SyncBatchNorm layers _can be used_ with `torch.compile` simultaneously. 
Unfortunately, due to implementation details the Sync BN have to break the model graph each time BN layer is encountered to perform BN sync operation.
That means you will most likely get no speedup from `torch.compile` when using Sync BN.
It also was observed that it may require more GPU memory compared to training without `torch.compile`.
If you are running into errors when using `torch.compile` and `SyncBatchNorm` simultaneously, you can try lowering the batch size to see if that helps.

#### Increased GPU memory consumption and Torch Compile

If during `torch.compile` you are getting wierd CUDA-related exception messages you can try reducing batch size. 
When using `reduce-overhead` or `max-autotune` modes peak GPU memory consumption may be higher compared to training without `torch.compile`.
It is good idea to run code with `CUDA_LAUNCH_BLOCKING=1` first as it usually provides more meaningful error messages.

#### Auto Mixed Precision and Torch Compile

Best speedup was achieved with combination of `torch.compile` and AMP enabled. For F32 training `torch.compile` may not provide any speedup at all.
This is highly dependent on the target GPU and support of fp32 tensor cores, so we leave it up to the user to decide whether to use AMP or not.

For AMP training with `torch.compile` enabled, you need to pass `mixed_precision: True` to training hyperparameters:
  ```yaml
  training_hyperparams:
    torch_compile: True 
    mixed_precision: True
  ```
