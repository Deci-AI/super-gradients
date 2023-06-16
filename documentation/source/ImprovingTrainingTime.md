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
Please note that the improvement may vary depending on the model architecture, task, and training hyperparameters.

| Task                    | Recipe                    | Time per epoch in seconds (Baseline)   | Time per epoch in seconds (torch.compile)   | Improvement, %   | Mode            |
|-------------------------|---------------------------|----------------------------------------|---------------------------------------------|------------------|-----------------|
| Semantic Segmentation   | cityscapes_pplite_seg75   | 270.63                                 | 119.11                                      | 56%              | Single GPU      |
| Semantic Segmentation   | cityscapes_pplite_seg75   | 49.95                                  | 35.91                                       | 18%              | DDP             |
| Semantic Segmentation   | cityscapes_regseg48       | 125.14                                 | 108.57                                      | 13.2%            | Single GPU      |
| Semantic Segmentation   | cityscapes_regseg48       | 44.959                                 | 44.55                                       | 0.9%             | DDP             |
| Semantic Segmentation   | cityscapes_segformer      | 199.97                                 | 162.52                                      | 18.7%            | Single GPU      |
| Semantic Segmentation   | cityscapes_segformer      | 46.21                                  | 43.71                                       | 5.4%             | DPP             |
| Semantic Segmentation   | cityscapes_stdc_seg75     | 425.19                                 | 153.16                                      | 63.9%            | Single GPU      |
| Semantic Segmentation   | cityscapes_stdc_seg75     | 73.07                                  | 45.89                                       | 37.19%           | DDP             |
| Semantic Segmentation   | cityscapes_ddrnet         | 226.51                                 | 174.11                                      | 23.1%            | Single GPU      |
| Semantic Segmentation   | cityscapes_ddrnet         | 51.78                                  | 48.29                                       | 7.3%             | DDP             |
| ----------------------- | ------------------------- | -------------------------------------- | ------------------------------------------- | ---------------- | --------------- |
| Object Detection        | coco2017_yolo_nas_s       |                                        |                                             |                  | Single GPU      |
| Object Detection        | coco2017_yolo_nas_s       | 384.41                                 | 376.10                                      | 2.42%            | DDP             |
| Object Detection        | coco2017_yolo_nas_m       |                                        |                                             |                  | Single GPU      |
| Object Detection        | coco2017_yolo_nas_m       |                                        | 508.40                                      |                  | DDP             |

In the table above, number are reported as speedup compared to the baseline training time. 
Both experiments were run on 8x 3090 GPUs using PyTorch 2.0 with CUDA 11.8. 
Training was done for 5 epochs and median value was picked to compute the speedup. 
All experiments conducted with mixed precision (AMP) enabled, and SyncBN and EMA disabled.

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
Additionally, some training features are not supported either. 
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