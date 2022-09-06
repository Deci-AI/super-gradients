from super_gradients.training.models.detection_models.yolox import YoloX_N, YoloX_T, YoloX_S
from deci_common.data_types.enum.models_enums import QuantizationLevel
from deci_common.data_types.enum.model_frameworks import FrameworkType
from deci_optimize.converter import Converter
import os
import hydra
import pkg_resources
from super_gradients.common.data_types import StrictLoad
from super_gradients.training import models


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
        onnx_simplifier=True,
        packaging_format='raw',
        verbose=True,
        model_params=model_params
    )


def compile_nn_module_with_batched_nms(module, batch_size, input_dims, output_path, framework=FrameworkType.TENSORRT):
    compile_nn_module(module, batch_size, input_dims, output_path, framework,
                      model_params={'detection_output_processing_method': 'trt_batched_nms', 'num_classes': 80})


def prep(model, input_shape):
    model.eval()
    model.prep_model_for_conversion(input_shape)
    return model


def get_model(arch_params_config_name):
    name= arch_params_config_name[:7]
    model = models.get(name=name,
                       num_classes=80,
                       arch_params=get_arch_parms(arch_params_config_name),
                       strict_load=StrictLoad.ON,
                       pretrained_weights="coco",
                       checkpoint_path=None,
                       load_backbone=True
                       )
    return model


def get_arch_parms(config_name: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    initialize_config_dir(pkg_resources.resource_filename("super_gradients.recipes", "arch_params"))
    cfg = compose(config_name=config_name)
    cfg = hydra.utils.instantiate(cfg)
    GlobalHydra.instance().clear()
    return cfg


if __name__ == '__main__':
    base_model_dir = f'/home/naveassaf/Desktop/NMS_Benchmarks/A4000_NEW'
    framework = FrameworkType.TENSORRT


    if os.path.exists(base_model_dir):
        raise ValueError(f'Directory: {base_model_dir} already exists! Exiting')
    else:
        os.mkdir(base_model_dir)

    print('--------------------------------- YOLOX SMALL ---------------------------------\n')
    os.mkdir(os.path.join(base_model_dir, 'yolox_s'))
    batch_size = 32
    input_shape = [batch_size, 3, 640, 640]
    # model = prep(YoloX_S_First100(HpmStruct()), input_shape)
    # compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_s', 'first_100.engine'),
    #                   framework=framework)
    # del model

    model = prep(get_model("yolox_s_arch_params"), input_shape)
    compile_nn_module_with_batched_nms(model, input_shape[0], input_shape[1:],
                                       os.path.join(base_model_dir, 'yolox_s', 'batched_nms.engine'), framework=framework)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_s', 'no_nms.engine'),
                      framework=framework)
    del model
    print('-----------------------------------------------------------------------------\n')


    print('--------------------------------- YOLOX TINY ---------------------------------\n')
    os.mkdir(os.path.join(base_model_dir, 'yolox_t'))
    batch_size = 64
    input_shape = [batch_size, 3, 416, 416]
    # model = prep(YoloX_T_First100(HpmStruct()), input_shape)
    # compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_t', 'first_100.engine'),
    #                   framework=framework)
    # del model

    model = prep(get_model("yolox_tiny_arch_params"), input_shape)
    compile_nn_module_with_batched_nms(model, input_shape[0], input_shape[1:],
                                       os.path.join(base_model_dir, 'yolox_t', 'batched_nms.engine'), framework=framework)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_t', 'no_nms.engine'),
                      framework=framework)
    del model
    print('-----------------------------------------------------------------------------\n')


    print('--------------------------------- YOLOX NANO ---------------------------------\n')
    os.mkdir(os.path.join(base_model_dir, 'yolox_n'))
    batch_size = 64
    input_shape = [batch_size, 3, 416, 416]
    # model = prep(YoloX_N_First100(HpmStruct()), input_shape)
    # compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_n', 'first_100.engine'),
    #                   framework=framework)
    # del model

    model = prep(get_model("yolox_nano_arch_params"), input_shape)
    compile_nn_module_with_batched_nms(model, input_shape[0], input_shape[1:],
                                       os.path.join(base_model_dir, 'yolox_n', 'batched_nms.engine'), framework=framework)
    compile_nn_module(model, input_shape[0], input_shape[1:], os.path.join(base_model_dir, 'yolox_n', 'no_nms.engine'),
                      framework=framework)
    del model
    print('-----------------------------------------------------------------------------\n')
