from deci_common.data_types.enum.models_enums import QuantizationLevel
from deci_common.data_types.enum.model_frameworks import FrameworkType
import sys
if len(sys.argv) != 5:
    print('USAGE: [PATH_TO_RT_OPTIMIZATION] [PATH_TO_SG_SRC] [PATH_TO_OUTPUT_DIR] [FRAMEWORK(trt/onnx)]')
    exit(1)
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])

from deci_optimize.converter import Converter
import os
import hydra
import pkg_resources
from super_gradients.common.data_types import StrictLoad
from super_gradients.training import models
from torch import nn


class Slicer(nn.Module):
    def __init__(self):
        super(Slicer, self).__init__()

    def forward(self, x):
        return x[0][:, :100, :]


class ModelBuilder:
    def __init__(self, input_shape, arch_params_config_name, target_framework, target_directory, verbose=False):
        self._input_shape = input_shape
        self._model_name = arch_params_config_name[:7]
        self._arch_params = self._get_arch_parms(arch_params_config_name)
        self._target_framework = target_framework
        self._base_sg_module = self._prep_module(self._get_sg_module())
        self._first_100_module = nn.Sequential(self._base_sg_module, Slicer().eval())
        self.target_directory = target_directory
        self._verbose = verbose

    def compile_nn_module(self, module, output_path, model_params=None):
        Converter.convert_framework_to_framework(
            source_model=module,
            source_framework=FrameworkType.PYTORCH,
            target_framework=self._target_framework,
            batch_size=self._input_shape[0],
            input_dims=tuple(self._input_shape[1:]),
            target_ckpt_full_path=output_path,
            quantization_level=QuantizationLevel.FP16 if self._target_framework == FrameworkType.TENSORRT else QuantizationLevel.FP32,
            onnx_opset_ver=15,
            dynamic_batch_size=True,
            onnx_simplifier=True,
            packaging_format='raw',
            verbose=self._verbose,
            model_params=model_params
        )

    def compile_nn_module_with_batched_nms(self, module, output_path):
        self.compile_nn_module(module, output_path,
                               model_params={
                                   'detection_output_processing_method': 'trt_batched_nms',
                                   'num_classes': 80
                               })

    def _prep_module(self, module):
        module.eval()
        module.prep_model_for_conversion(self._input_shape)
        return module

    def _get_sg_module(self):
        model = models.get(name=self._model_name,
                           num_classes=80,
                           arch_params=self._arch_params,
                           strict_load=StrictLoad.ON,
                           pretrained_weights="coco",
                           checkpoint_path=None,
                           load_backbone=True
                           )
        return model

    @staticmethod
    def _get_arch_parms(config_name: str):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        initialize_config_dir(pkg_resources.resource_filename("super_gradients.recipes", "arch_params"))
        cfg = compose(config_name=config_name)
        cfg = hydra.utils.instantiate(cfg)
        GlobalHydra.instance().clear()
        return cfg

    def build_all_models(self, output_dir=None):
        output_dir = output_dir or os.path.join(self.target_directory, self._model_name)
        postfix = 'onnx' if self._target_framework == FrameworkType.ONNX else 'engine'

        os.mkdir(output_dir)

        print(f'\n\nCOMPILING {self._model_name} - NO NMS\n\n')
        self.compile_nn_module(self._base_sg_module, os.path.join(output_dir, f'no_nms.{postfix}'))

        print(f'\n\nCOMPILING {self._model_name} - BATCHED NMS\n\n')
        self.compile_nn_module_with_batched_nms(self._base_sg_module,
                                                os.path.join(output_dir, f'batched_nms.{postfix}'))

        print(f'\n\nCOMPILING {self._model_name} - FIRST 100\n\n')
        self.compile_nn_module(self._first_100_module, os.path.join(output_dir, f'first_100.{postfix}'))


if __name__ == '__main__':
    base_model_dir = sys.argv[3]
    target_framework = FrameworkType.TENSORRT if sys.argv[4].lower() == 'trt' else FrameworkType.ONNX

    if os.path.exists(base_model_dir):
        raise ValueError(f'Directory: {base_model_dir} already exists! Exiting')
    else:
        os.mkdir(base_model_dir)

    print('--------------------------------- YOLOX SMALL ---------------------------------\n')
    yoloxs_builder = ModelBuilder(input_shape=[32, 3, 640, 640],
                                  arch_params_config_name="yolox_s_arch_params",
                                  target_framework=target_framework,
                                  target_directory=base_model_dir)
    yoloxs_builder.build_all_models()
    del yoloxs_builder
    print('-----------------------------------------------------------------------------\n')

    print('--------------------------------- YOLOX TINY ---------------------------------\n')
    yoloxt_builder = ModelBuilder(input_shape=[64, 3, 416, 416],
                                  arch_params_config_name="yolox_tiny_arch_params",
                                  target_framework=target_framework,
                                  target_directory=base_model_dir)
    yoloxt_builder.build_all_models()
    del yoloxt_builder
    print('-----------------------------------------------------------------------------\n')

    print('--------------------------------- YOLOX NANO ---------------------------------\n')
    yoloxn_builder = ModelBuilder(input_shape=[64, 3, 416, 416],
                                  arch_params_config_name="yolox_nano_arch_params",
                                  target_framework=target_framework,
                                  target_directory=base_model_dir)
    yoloxn_builder.build_all_models()
    del yoloxn_builder
    print('-----------------------------------------------------------------------------\n')
