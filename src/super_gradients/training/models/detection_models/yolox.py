from super_gradients.training.models.detection_models.yolo_base import YoLoBase, YoLoDarknetBackbone
import numpy as np
import torch
from super_gradients.training.models.detection_models.yolov5_base import YoLoV5Base, YoLoV5DarknetBackbone
from super_gradients.training.utils.utils import HpmStruct

import sys
import os
sys.path.insert(0, '/home/naveassaf/Workspace/rt-optimization')

from deci_common.data_types.enum.models_enums import QuantizationLevel
from deci_common.data_types.enum.model_frameworks import FrameworkType
from deci_optimize.converter import Converter

# ------------------------------------------------------------------------------------------ #
# YOLOX Nano
# ------------------------------------------------------------------------------------------ #
class YoloX_N(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.25
        arch_params.yolo_type = 'yoloX'
        arch_params.depthwise = True
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_N_First100(YoloX_N):
    def forward(self, x):
        return super().forward(x)[0][:, :100, :]

# ------------------------------------------------------------------------------------------ #
# YOLOX Tiny
# ------------------------------------------------------------------------------------------ #
class YoloX_T(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.375
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_T_First100(YoloX_T):
    def forward(self, x):
        return super().forward(x)[0][:, :100, :]

# ------------------------------------------------------------------------------------------ #
# YOLOX Small
# ------------------------------------------------------------------------------------------ #
class YoloX_S(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.50
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_S_First100(YoloX_S):
    def forward(self, x):
        return super().forward(x)[0][:, :100, :]


# ------------------------------------------------------------------------------------------ #
class YoloX_M(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.67
        arch_params.width_mult_factor = 0.75
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_L(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.0
        arch_params.width_mult_factor = 1.0
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_X(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.33
        arch_params.width_mult_factor = 1.25
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


def compile_nn_module(module, batch_size, input_dims, output_path, framework=FrameworkType.TENSORRT, model_params=None):
    Converter.convert_framework_to_framework(
        source_model=module,
        source_framework=FrameworkType.PYTORCH,
        target_framework=framework,
        batch_size=batch_size,
        input_dims=tuple(input_dims),
        target_ckpt_full_path=output_path,
        quantization_level=QuantizationLevel.FP16 if framework == FrameworkType.TENSORRT else QuantizationLevel.FP32,
        onnx_opset_ver=15,
        dynamic_batch_size=True,
        onnx_simplifier=False,
        packaging_format='raw',
        verbose=False,
        model_params=model_params
    )


def compile_nn_module_with_batched_nms(module, batch_size, input_dims, output_path, framework=FrameworkType.TENSORRT):
    compile_nn_module(module, batch_size, input_dims, output_path, framework,
                      model_params={'detection_output_processing_method': 'trt_batched_nms', 'num_classes': 80})


def prep(model, input_shape):
    model.eval()
    model.prep_model_for_conversion(input_shape)
    return model


if __name__ == '__main__':
    base_model_dir = f'/home/naveassaf/Desktop/NMS_Benchmarks/A4000'
    framework = FrameworkType.TENSORRT


    if os.path.exists(base_model_dir):
        raise ValueError(f'Directory: {base_model_dir} already exists! Exiting')
    else:
        os.mkdir(base_model_dir)

    print('--------------------------------- YOLOX SMALL ---------------------------------\n')
    os.mkdir('yolox_s')
    batch_size = 32
    input_shape = [batch_size, 3, 640, 640]
    model = prep(YoloX_S_First100(HpmStruct()), input_shape)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_s', 'first_100.engine'),
                      framework=framework)
    del model

    model = prep(YoloX_S(HpmStruct()), input_shape)
    compile_nn_module_with_batched_nms(model, input_shape[0], input_shape[1:],
                                       os.path.join(base_model_dir, 'yolox_s', 'batched_nms.engine'), framework=framework)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_s', 'no_nms.engine'),
                      framework=framework)
    del model
    print('-----------------------------------------------------------------------------\n')


    print('--------------------------------- YOLOX TINY ---------------------------------\n')
    batch_size = 64
    input_shape = [batch_size, 3, 416, 416]
    model = prep(YoloX_T_First100(HpmStruct()), input_shape)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_t', 'first_100.engine'),
                      framework=framework)
    del model

    model = prep(YoloX_T(HpmStruct()), input_shape)
    compile_nn_module_with_batched_nms(model, input_shape[0], input_shape[1:],
                                       os.path.join(base_model_dir, 'yolox_t', 'batched_nms.engine'), framework=framework)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_t', 'no_nms.engine'),
                      framework=framework)
    del model
    print('-----------------------------------------------------------------------------\n')


    print('--------------------------------- YOLOX NANO ---------------------------------\n')
    batch_size = 64
    input_shape = [batch_size, 3, 416, 416]
    model = prep(YoloX_N_First100(HpmStruct()), input_shape)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_n', 'first_100.engine'),
                      framework=framework)
    del model

    model = prep(YoloX_N(HpmStruct()), input_shape)
    compile_nn_module_with_batched_nms(model, input_shape[0], input_shape[1:],
                                       os.path.join(base_model_dir, 'yolox_n', 'batched_nms.engine'), framework=framework)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_n', 'no_nms.engine'),
                      framework=framework)
    del model
    print('-----------------------------------------------------------------------------\n')
