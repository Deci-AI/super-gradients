# Installing SuperGradients

## Requirements

### General requirements

  
- Python 3.7, 3.8 or 3.9 installed.
- torch>=1.9.0
  - https://pytorch.org/get-started/locally/
- The python packages that are specified in requirements.txt;

  
### To train on nvidia GPUs
  
- [Nvidia CUDA Toolkit >= 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu)
- CuDNN >= 8.1.x
- Nvidia Driver with CUDA >= 11.2 support (≥460.x)

## Quick Installation

### Install stable version using PyPi 

See in [PyPi](https://pypi.org/project/super-gradients/)
```bash
pip install super-gradients
```

That's it !

> **Important**: If PyTorch was not already installed on your environment, you might need to reinstall
> a Pytorch version suitable for your CUDA version. Go into [PyTorch installation page](https://pytorch.org/get-started/locally/)
> and follow the instructions to install the correct version.  


### Install using GitHub</summary>


```bash
pip install git+https://github.com/Deci-AI/super-gradients.git@stable
```
