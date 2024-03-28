# OpenVINO Integration to SG

# Done / Know how to be done

- We can do PTQ (quantize_from_recipe) of YoloNAS. It sort of works, but the quality drop is significant:
    
    ```yaml
    YoloNAS-S
    47.5 (FP16)
    47.03 (INT8) - TRT (As reported in model_zoo.md)
    42.75 (INT8) - OpenVINO w/o selective
    42.85 (INT8) - OpenVINO with selective
    
     
    ```
    

# Don’t know how to do (yet?)

- It is not clear how to attach postprocessing to a quantized OpenVINO model. Is it possible to export quantized OpenVINO model to ONNX and attach postprocessing ONNX graph to it?
- QAT possibility was not researched
- Off-the shelf quantization parameters gives a poor PTQ results (47 -> 42 mAP). We don't know what are the optimal quantization settings (Maybe it is even model-specific). A grid-search is needed.
- PTQ with quality control does not support Pytorch models. Can be done with OpenVINO models but then it's quite unclear would it be possible afterwards to attach postprocessing to it.

  

# TODO


## Fallback to cpu if CUDA toolkit is not present

OpenVINO can quantize models that are on `cpu` and `gpu` devices. However for `gpu` it needs `nvcc` and CUDA toolkit installed and accessible through `PATH` in order to compile GPU quantization extensions. If CUDA toolkit is not available and model is on `gpu`  OpenVINO will raise `nncf.errors.InstallationError`  We can intercept that and either move model to `cpu` or raise a more informative error message

## Setting layers for selective quantization

Performing a selective quantization is possible, but very non-intuitive.

Here is how one can disable quantization of certain layers using SelectiveQuantizer:
```yaml
selective_quantizer_params:
  skip_modules:              # optional list of module names (strings) to skip from quantization
    - heads.head1.reg_convs
    - heads.head1.cls_convs
    - heads.head1.cls_pred
    - heads.head1.reg_pred

    - heads.head2.reg_convs
    - heads.head2.cls_convs
    - heads.head2.cls_pred
    - heads.head2.reg_pred

    - heads.head3.reg_convs
    - heads.head3.cls_convs
    - heads.head3.cls_pred
    - heads.head3.reg_pred
```

For OpenVINO the naming scheme is completely different and refers to a NNCF graph

```yaml
selective_quantizer_params:
  skip_modules:              # optional list of module names (strings) to skip from quantization
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/Sequential[cls_convs]
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/Sequential[cls_convs]
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/Sequential[cls_convs]

    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/Sequential[reg_convs]
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/Sequential[reg_convs]
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/Sequential[reg_convs]

    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/NNCFConv2d[cls_pred]
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/NNCFConv2d[cls_pred]
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/NNCFConv2d[cls_pred]

    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/NNCFConv2d[reg_pred]
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/NNCFConv2d[reg_pred]
    - YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/NNCFConv2d[reg_pred]
```

Probably there is a strict rule how NNCF gives names to these variables, and maybe there is an utility function one can use to generate corresponding names.
But the only option to get these names I've found so far was to export model w/o selective quantization, print NNCF graph and pick layer names from it.
Clearly, manual picking of layer names it's not production-ready approach and this should be addressed somehow.

There is a way to use regular expressions which should look much simplier: `heads.+reg_pred` but this hasn't been tested yet
