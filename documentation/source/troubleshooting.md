# Troubleshooting

This tutorial addresses some of the most frequent concerns we've seen.

If you want more assistance in solving your problem, you may open a new 
[Issue](https://github.com/Deci-AI/super-gradients/issues/new?assignees=&labels=&template=bug_report.md&title=) 
in the SuperGradients repository.




## CUDA Version error

When using SuperGradients for the first time, you might get this error;
```
OSError: .../lib/python3.8/site-packages/nvidia/cublas/lib/libcublas.so.11: undefined symbol: cublasLtGetStatusString, version libcublasLt.so.11
```

This may indicate a CUDA conflict between libraries (When Torchvision & Torch are installed for different CUDA versions) or the absence of CUDA support in your Torch version.
To fix this you can

- Uninstall both torch and torchvision `pip unistall torch torchvision`
- Install the torch version that respects your **os** & **compute platform** following the instruction from https://pytorch.org/



## GPU Memory Overflow

It is pretty common to run out of memory when using GPU. This is shown with following exception:

```
CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 10.76 GiB total capacity; 4.29 GiB already allocated; 10.12 MiB free; 4.46 GiB reserved in total by PyTorch)
```

To reduce memory usage, try the following

- Decrease the batch size (`dataset_params.train_dataloader_params.batch_size` and `dataset_params.val_dataloader_params.batch_size`)
- Adjust the number of batch accumulation steps (`training_hyperparams.batch_accumulate`) and/or number of nodes (if you are using [DDP](device.md)) to keep the effective batch size the same: `effective_batch_size = num_gpus * batch_size * batch_accumulate` 


## CUDA error: device-side assert triggered

You may encounter a generic CUDA error message that lacks information regarding the cause of the error:

```
RuntimeError: CUDA error: device-side assert triggered
```

To get a better understanding of the root cause of the error, you have the choice between two approaches:

**1. Run on CPU**

When [running on CPU](device.md) you won't have this issue of CUDA hiding the root cause of the error.

**2. Set Environment Variable**

Some environment variables can be helpful in identifying the root cause:

- `CUDA_LAUNCH_BLOCKING=1` can be used to force synchronous execution of kernel launches, allowing you to pinpoint the exact location of the error in your code.
- `CUDA_DEVICE_ASSERT=1` can be used to enable detailed error messages that provide the file name and line number where the assert was triggered.
