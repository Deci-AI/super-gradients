# Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)

### Content

* Introduction
* Quantization: FP32 vs FP16 vs INT8
* Post-training Quantization
* Quantization-Aware training
* Converting quantized models to ONNX for inference
* Using SuperGradient's Recipes for PTQ/QAT

## Introduction

As deep learning models have grown in their complexity and applications,
they’ve also grown large and cumbersome. Large models running on cloud environments have huge compute demand resulting in high cloud cost for developers, posing a major barrier for profitability and scalability. For edge deployments, edge devices are resource-constrained and therefore can not support large and complex models. 

Whether the model is deployed on the cloud or at the edge, AI developers are often confronted with the challenge of reducing their model size without compromising model accuracy. Quantization is a common technique used to reduce model size, though it can sometimes result in reduced accuracy. 

Quantization aware training is a method that allows practitioners to apply quantization techniques without sacrificing accuracy. It is done in the model training process rather than after the fact. The model size can typically be reduced by two to four times, and sometimes even more. 

In this tutorial, we’ll compare post-training quantization (PTQ) to quantization-aware training (QAT), and demonstrate how both methods can be easily performed using Deci’s SuperGradients library.

For mode detailed information and theoretical background, refer to this [NVIDIA whitepaper](https://arxiv.org/pdf/2004.09602.pdf) and [this practical guide from PyTorch](https://pytorch.org/blog/quantization-in-practice/).


## Quantization: FP32 vs FP16 vs INT8
Quantization is a model size reduction technique that converts model weights from high-precision floating-point representation (32-bit float) to low-precision floating-point (FP) representation, such as 16-bit or 8-bit.

During quantization, the dynamic range of the original high-precision model has to be compressed into a limited range of the low-precision representation. To achieve this, a calibration process is employed to determine the minimum, maximum, and scale parameters that map the high-precision representation to the low-precision. 

The calibration process is performed using a set of representative data samples, known as the calibration dataset, to ensure that the quantization process preserves the model's accuracy as much as possible. The most commonly supported calibration methods are percentile, max, and entropy, which are available in most deep learning frameworks. By using these methods, the quantization process can adapt the parameters based on the specific characteristics of the model and the calibration dataset, resulting in a more accurate quantized model.

## Post-training Quantization

Post-training quantization (PTQ) is a quantization method where the quantization process is applied to the trained model after it has completed training. The model's weights and activations are quantized from high precision to low precision, such as from FP32 to INT8. This method is simple and straightforward to implement, but it does not account for the impact of quantization during the training process.

### Hybrid quantization
__PREREQUISITE__: You will need `pytorch_quantization` installed:

```shell
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```
With SuperGradients, performing hybrid quantization takes just two lines of code, except of the model definition:

```python
import super_gradients.training.models
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

model = super_gradients.training.models.get(model_name="resnet50", pretrained_weights="imagenet")

q_util = SelectiveQuantizer(
    default_quant_modules_calibrator_weights="max",
    default_quant_modules_calibrator_inputs="histogram",
    default_per_channel_quant_weights=True,
    default_learn_amax=False,
    verbose=True,
)
q_util.quantize_module(model)
```

### Selective quantization
SuperGradients supports selective and partial quantization: skipping modules from quantization, or replacing them with quantization-friendly counterparts. 

Using the API of `SelectiveQuantizer` it is straightforward, and it offers great flexibility:

You can skip modules by their names, or by their types:
```python
from torch import nn

q_util.register_skip_quantization(layer_names={
    "layer1",
    "layer2.0.conv1",
    "conv1"
})

q_util.register_skip_quantization(layer_names={nn.Linear})
```

You can replace modules with another type, e.g. replace `Bottleneck` with SuperGradients' `QuantBottleneck`:

```python
from super_gradients.training.models import Bottleneck
from super_gradients.modules.quantization import QuantBottleneck

q_util.register_quantization_mapping(layer_names={Bottleneck},
                                     quantized_target_class=QuantBottleneck,
                                     input_quant_descriptor=QuantDescriptor(...),
                                     weights_quant_descriptor=QuantDescriptor(...))
```

Additionally, if you are designing your own custom block, you can register it, so it will be automatically used for replacement:

```python
from super_gradients.training.utils.quantization.selective_quantization_utils import register_quantized_module
from super_gradients.training.utils.quantization.selective_quantization_utils import QuantizedMetadata

@register_quantized_module(float_source=MyNonQuantBlock,
                           action=QuantizedMetadata.ReplacementAction.REPLACE,
                           input_quant_descriptor=QuantDescriptor(...),
                           weights_quant_descriptor=QuantDescriptor(...)
                           )
class MyQuantBlock:
    ...
```

#### QuantDescriptor API

`QuantDescriptor` is a class that is used to configure `TensorQuantizer` for weights and activations. This class if from `pytorch-quantization` library and has the following API:

```    
Args:
    num_bits: An integer. Number of bits of quantization. It is used to calculate scaling factor. Default 8.
    name: Seems a nice thing to have

Keyword Arguments:
    fake_quant: A boolean. If True, use fake quantization mode. Default True.
    axis: None, int or tuple of int. axes which will have its own max for computing scaling factor.
        If None (the default), use per tensor scale.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
        e.g. For a KCRS weight tensor, quant_axis=(0) will yield per channel scaling.
        Default None.
    amax: A float or list/ndarray of floats of user specified absolute max range. If supplied,
        ignore quant_axis and use this to quantize. If learn_amax is True, will be used to initialize
        learnable amax. Default None.
    learn_amax: A boolean. If True, learn amax. Default False.
    scale_amax: A float. If supplied, multiply amax by scale_amax. Default None. It is useful for some
        quick experiment.
    calib_method: A string. One of ["max", "histogram"] indicates which calibration to use. Except the simple
        max calibration, other methods are all hisogram based. Default "max".
    unsigned: A Boolean. If True, use unsigned. Default False.
```
Use it to customize your flow. It is recommended to leave default values at least for early experiments. 


### Quantizing residuals and skip connections
To improve performance of quantized models, quantization of residuals and skip connections is performed. SuperGradients API allows you to do it. In your source code, add one of the following, depending on the type of the skip connection: 

```python
from super_gradients.modules.skip_connections import (
    Residual, 
    SkipConnection, 
    CrossModelSkipConnection, 
    BackboneInternalSkipConnection, 
    HeadInternalSkipConnection
)
```

Use them for all inputs of the `sum`, `mul`, `div` and `concat` operations. `SelectiveQuantizer` will take care of them and will replace them with quantized counterparts.


For example, take a simple resnet-like block:

```python
from torch import nn
import torch.nn.functional as F

class ResNetLikeBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResNetLikeBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = F.relu(out + x)

        return out
```

Its quantizeable modification will look like this:

```python
from torch import nn
import torch.nn.functional as F
from super_gradients.modules.skip_connections import Residual

class ResNetLikeBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResNetLikeBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.residual = Residual()
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        res = self.residual(x)
        
        out = F.relu(out + res)

        return out
```


### Calibration
And after quantization, performing calibration take another two lines of code:

```python
from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
model = ... # your quantized model
calib_dataloader = ... # your standard pytorch dataloader

calibrator = QuantizationCalibrator(verbose=True)
calibrator.calibrate_model(
    model,
    method="percentile",
    calib_data_loader=calib_dataloader,
    num_calib_batches=16,
    percentile=99.99,
)
```

Your model is now quantized and calibrated!

Refer to `super_gradients/src/super_gradiens/examples/quantization` for more source examples that are ready-to-run!


## Quantization-Aware training

Quantization-aware training (QAT) is a method that takes into account the impact of quantization during the training process. The model is trained with quantization-aware operations that mimic the quantization process during training. This allows the model to learn how to perform well in the quantized representation, leading to improved accuracy compared to post-training quantization.

With SuperGradients, after you have done PTQ, you can finetune your quantized model with standard training pipeline:

```python
from super_gradients import Trainer

model = ...  # your quantized and calibrated model
train_dataloader = ... # your standard pytorch dataloader
valid_dataloader = ... # your standard pytorch dataloader
training_hyperparams = ... # refer to training_hyperparams example to fill it

model.train()
trainer = Trainer(experiment_name="my_first_qat_experiment", ckpt_root_dir=...)
res = trainer.train(
    model=model,
    train_loader=train_dataloader,
    valid_loader=valid_dataloader,
    training_params=training_hyperparams
)

```

After that, your model will be finetuned with quantization in mind!

## Converting quantized models to ONNX for inference

SG is a Production ready library. All the models implemented in SG can be compiled to ONNX, even quantized ones.

If you are using a recipe, neatly quantized ONNX will wait for you in the checkpoints directory. 
If you prefer more of a DIY approach, here is the code sample:

```python
import torch
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx

onnx_filename = f"qat_model_1x3x224x224.onnx"

dummy_input = torch.randn([1, 3, 224, 224], device="cpu")
export_quantized_module_to_onnx(
    model=quantized_model.cpu(),
    onnx_filename=onnx_filename,
    input_shape=[1, 3, 224, 224],
    input_size=[1, 3, 224, 224],
    train=False,
)
```

Note that this ONNX uses fake quantization (refer to ONNX `QuantizeLinear/DequantizeLinear` for more info), while being in FP32 itself. To get a quantized model, you will need an inference framework that will compile ONNX into a runnable engine. Here is an example how to do it with NVIDIA's TensorRT:

```shell
trtexec --int8 --fp16 --onnx=qat_model_1x3x224x224.onnx --saveEngine=qat_model_1x3x224x224.pkl
```
## Using SuperGradient's Recipes for PTQ/QAT

The SuperGradient library provides a simple and easy-to-use API for both post-training quantization and quantization-aware training. By using the library's recipes, you can quickly and easily quantize models without having to write custom code.

**Use `src/super_gradients/examples/qat_from_recipe_example/qat_from_recipe.py` to launch your QAT recipes, using `train_from_recipe.py` will lead you to wrong results!**

To get a basic understanding of recipes, refer to `configuration_files.md` for more details. 

You can modify an existing recipe to suit PTQ and QAT by adding `quantization_params` to it. You can find these `default_quantization_params` in  `src/super_gradients/recipes/quantization_params/default_quantization_params.yaml`

Also, you can add a sepatare calibration dataloader to your recipe , otherwise, train dataloader without augmenttations will be used for calibration:

```yaml
calib_dataloader: imagenet_train # for example
dataset_params:
  ...
  calib_dataloader_params:
    ...
  calib_dataset_params:
    ...
```

Initialization and parameters are identical to training and validation datasets and dataloaders. Refer to `configuration_files.md` for details.

```yaml
ptq_only: False              # whether to launch QAT, or leave PTQ only
selective_quantizer_params:
  calibrator_w: "max"        # calibrator type for weights, acceptable types are ["max", "histogram"]
  calibrator_i: "histogram"  # calibrator type for inputs acceptable types are ["max", "histogram"]
  per_channel: True          # per-channel quantization of weights, activations stay per-tensor by default
  learn_amax: False          # enable learnable amax in all TensorQuantizers using straight-through estimator
  skip_modules:              # optional list of module names (strings) to skip from quantization

calib_params:
  histogram_calib_method: "percentile"  # calibration method for all "histogram" calibrators, acceptable types are ["percentile", "entropy", mse"], "max" calibrators always use "max"
  percentile: 99.99                     # percentile for all histogram calibrators with method "percentile", other calibrators are not affected
  num_calib_batches:                    # number of batches to use for calibration, if None, 512 / batch_size will be used
  verbose: False                        # if calibrator should be verbose

```

As we have seen earlier, these are the same parameters in the YAML form. 

If you want to use our rules of thumb to modify your existing training recipe parameters for QAT, you need to use `QATRecipeModificationCallback`. To do it, add following config to your recipe:

```yaml
pre_launch_callbacks_list:
    - QATRecipeModificationCallback:
        batch_size_divisor: 2
        max_epochs_divisor: 10
        lr_decay_factor: 0.01
        warmup_epochs_divisor: 10
        cosine_final_lr_ratio: 0.01
        disable_phase_callbacks: True
        disable_augmentations: False
```

Default parameters of this callback are representing the rules of thumb to perform successful QAT from an existing training recipe. 
