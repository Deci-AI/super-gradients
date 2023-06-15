# Improving Training Time

## Mixed Precision Training

Automatic mixed precision (AMP) is a feature in PyTorch that enables the use of lower-precision data types, such as float16, in deep learning models for improved memory and computation efficiency. 
It automatically casts the model's parameters and buffers to a lower-precision data type, and dynamically rescales the activations to prevent underflow or overflow. 

Most modern GPUs [support](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix) float16 operations natively, and can therefore accelerate the training process.

To use `AMP` in SuperGradients, you simply need to set `mixed_precision=True` in your training_params.

**In python script**
```python
from super_gradients import Trainer

trainer = Trainer("experiment_name")
model = ...

training_params = {"mixed_precision": True, ...:...}

trainer.train(model=model, training_params=training_params, ...)
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
To leverage use of compiled models in SuperGradients one need to pass the `torch_compile: True` option to training hyperparameters:

```bash
python -m super_gradients.train_from_recipe --config-name=... training_params.torch_compile=True
```

In the YAML recipe:

```yaml
# my_training_recipe.yaml
training_params:
  torch_compile: True 
  torch_compile_mode: default | reduce-overhead | max-autotune
```


Or programmatically:

```python
from super_gradients.training import Trainer

trainer = Trainer(
    ...,
    training_params = {
        "torch_compile": True,
        ...
        }
    )
```



### Avoiding common pitfalls:

Torch Compile is still in its early stages and has some limitations. Not every model can be compiled. 
Additionally, some training features are not supported either. 
Here is what we found so far:

* Exponential moving average `EMA` is incompatible with `torch.compile`. 
  If you want to use `torch.compile` in your training, you need to disable `EMA` when using `torch.compile`.
  ```yaml
  training_params:
    torch_compile: True 
    ema: False
  ```
  
#### Torch Compile and Sync BatchNorm

When training using DDP, SyncBatchNorm layers can be used with `torch.compile` simultaneously. 
Unfortunately, due to implementation details the Sync BN makes compline to break the model graph each time that BN synchronization is performed.
That means you will most likely get no speedup from `torch.compile` when using Sync BN.
It also was observed that it may require more GPU memory compared to training without `torch.compile`.
If you are running into errors when using `torch.compile` and `SyncBatchNorm` simultaneously, you can try lowering the batch size to see if that helps.


* If during `torch.compile` you are getting wierd CUDA-related exception messages () you can try reducing batch size. 
  When using `reduce-overhead` or `max-autotune` modes peak GPU memory consumption may be higher compared to training without `torch.compile`.
  It is good idea to run code with `CUDA_LAUNCH_BLOCKING=1` first as it usually provides more meaningful error messages.

* Best speedup was achieved with combination of `torch.compile` + AMP. For F32 training `torch.compile` may not provide any speedup at all.
  ```yaml
  training_params:
    torch_compile: True 
    mixed_precision: True
  ```
