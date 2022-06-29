"""
Quantization utilities

Methods are based on:
 https://github.com/NVIDIA/TensorRT/blob/51a4297753d3e12d0eed864be52400f429a6a94c/tools/pytorch-quantization/examples/torchvision/classification_flow.py#L385

(Licensed under the Apache License, Version 2.0)
"""

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.callbacks import Phase, PhaseCallback, PhaseContext
import os
import pickle
import sys
from enum import Enum

from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
from super_gradients.training.utils.distributed_training_utils import wait_for_the_master, get_local_rank

logger = get_logger(__name__)

try:
    import tensorrt as trt

    _imported_trt_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_trt_failure = import_err

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


class QuantizationLevel(str, Enum):
    FP32 = 'FP32'
    FP16 = 'FP16'
    INT8 = 'INT8'
    HYBRID = 'Hybrid'

    @staticmethod
    def from_string(quantization_level: str) -> Enum:
        quantization_level = quantization_level.lower()
        if quantization_level == 'fp32':
            return QuantizationLevel.FP32
        elif quantization_level == 'fp16':
            return QuantizationLevel.FP16
        elif quantization_level == 'int8':
            return QuantizationLevel.INT8
        elif quantization_level == 'hybrid':
            return QuantizationLevel.HYBRID
        else:
            raise NotImplementedError(f'Quantization Level: "{quantization_level}" is not supported')


def export_qat_onnx(model: torch.nn.Module, onnx_filename: str, input_shape: tuple):
    """
    Method for exporting onnx after QAT.

    :param model: torch.nn.Module, model to export
    :param onnx_filename: str, target path for the onnx file,
    :param input_shape: tuple, input shape (usually BCHW)
    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    else:
        model.eval()
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        # Export ONNX for multiple batch sizes
        logger.info("Creating ONNX file: " + onnx_filename)
        dummy_input = torch.randn(input_shape, device='cuda')
        torch.onnx.export(model, dummy_input, onnx_filename, verbose=False, opset_version=13, enable_onnx_checker=False,
                          do_constant_folding=True)


def calibrate_model(model: torch.nn.Module, calib_data_loader: torch.utils.data.DataLoader, method:str="percentile", num_calib_batches: int=2, percentile:float=99.99):
    """
    Calibrates torch model with quantized modules.

    :param model: torch.nn.Module, model to perfrom the calibration on.
    :param calib_data_loader: torch.utils.data.DataLoader, data loader of the calibration dataset.
    :param method: str, One of [percentile, mse, entropy, max]. Statistics method for amax
                 computation of the quantized modules (default=percentile).
    :param num_calib_batches: int, number of batches to collect the statistics from.
    :param percentile: float, percentile value to use when SgModel,quant_modules_calib_method='percentile'. Discarded when other methods are used (Default=99.99).

    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    elif method in ["percentile", "mse", "entropy", "max"]:
        with torch.no_grad():
            _collect_stats(model, calib_data_loader, num_batches=num_calib_batches)

            # FOR PERCENTILE WE MUST PASS PERCENTILE VALUE THROUGH KWARGS,
            # SO IT WOULD BE PASSED TO module.load_calib_amax(**kwargs), AND IN OTHER METHODS WE MUST NOT PASS IT.
            if method == "precentile":
                _compute_amax(model, method="percentile", percentile=percentile)
            else:
                _compute_amax(model, method=method)
    else:
        raise ValueError(
            "Unsupported quantization calibration method, expected one of: percentile, mse, entropy, max, got " + str(
                method) + ".")


def _collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    else:
        # Enable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
            model(image.cuda())
            if i >= num_batches:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()


def _compute_amax(model, **kwargs):
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    else:
        # Load calib result
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
                print(F"{name:40}: {module}")
        model.cuda()


def build_trt_engine_from_onnx_ckpt(onnx_ckpt_path: str,
                                    trt_max_batch_size: int,
                                    quantization_level: QuantizationLevel = QuantizationLevel.FP32):
    """
    A function for building a trt.ICudaEngine graph from an ONNX model.
    :param onnx_ckpt_path: Path to ONNX model.
    :param quantization_level: The precision to use. Currently supported FP32 and FP16.
    :param trt_max_batch_size: The max batch size allowed for inference.
    :return: An ICudaEngine object (graph of the model).
    """
    if _imported_trt_failure is not None:
        raise _imported_trt_failure
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)  # VERBOSE for printing
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            # Fill network attributes with information by parsing model
            with open(onnx_ckpt_path, "rb") as f:
                # Parse model and capture its exit status
                parse_success = parser.parse(f.read())
                # Catch any errors thrown while parsing and exit gracefully on failure
                if not parse_success:
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    sys.exit(1)

            # Query input names and shapes from parsed TensorRT network
            network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
            input_names = [_input.name for _input in network_inputs]  # ex: ["actual_input1"]
            # Note the original model must have dynamic (-1) dimensions for variable min/opt/max values
            # in the profile dimensions (such as the batch dimension)
            input_shapes = [_input.shape for _input in network_inputs]  # ex: [(-1, 3, 224, 224)]

            # Create optimization profile for dynamic batch dimension
            # Note optimal performance is set for max batch size
            profile0 = builder.create_optimization_profile()
            for name, shape in zip(input_names, input_shapes):
                profile0.set_shape(
                    name, min=(1, *shape[1:]), opt=(trt_max_batch_size, *shape[1:]),
                    max=(trt_max_batch_size, *shape[1:])
                )
            config.add_optimization_profile(profile0)

            # Additional builder_config flags can be set prior to building the engine
            if quantization_level == QuantizationLevel.FP16:
                config.set_flag(trt.BuilderFlag.FP16)
                try:
                    builder.fp16_mode = True
                except:
                    # TRT8 breaking API - supporting older versions
                    pass
            elif quantization_level == QuantizationLevel.INT8:
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            # Compilation parameters
            config.max_workspace_size = 4 << 30
            try:
                builder.max_workspace_size = 4 << 30
            except:
                # TRT8 breaking API - supporting older versions
                pass

            # Set max batch size
            builder.max_batch_size = trt_max_batch_size

            engine = builder.build_engine(network, config)

            return engine


def save_trt_engine_from_onnx_ckpt(onnx_ckpt_path, output_engine_path, quantization_level=QuantizationLevel.INT8,
                                   max_batch_size=1):
    if os.path.exists(onnx_ckpt_path):
        print("Building engine from file {}".format(onnx_ckpt_path))
        trt_engine = build_trt_engine_from_onnx_ckpt(onnx_ckpt_path=onnx_ckpt_path,
                                                     trt_max_batch_size=max_batch_size,
                                                     quantization_level=quantization_level)
        serialized_engine = bytes(trt_engine.serialize())
        engine_dict = {'engine': serialized_engine,
                       'compiler': 'trt',
                       'precision': quantization_level.value}
        with open(output_engine_path, "wb") as f:
            pickle.dump(engine_dict, f)
            print(
                f'Successfully converted {onnx_ckpt_path} to {output_engine_path} with max batch size {max_batch_size} and {quantization_level} quantization.')
    else:
        raise ValueError("The input file does not exist.")


class Int8CalibrationPreTrainingCallback(PhaseCallback):
    """
    Pre-training callback for calibrating model with Q/DQ modules.
    Saves the calibrated model using context.sg_logger.

    Note: calibration method is according to quant_modules_calib_method passed in arch_params in SgModel.build_model().

    Attributes:
        calib_data_loader: DataLoader,  dataloader to apply calibration on. When none, context.valid_loader will be used. (Default=None).
        num_calib_batches: int, number of batches to collect the statistics from (default=2).
        percentile: float, percentile value to use when SgModel,quant_modules_calib_method='percentile'. Discarded when other methods are used (Default=99.99).

        TODO: MAKE DDP SAFE
    """

    def __init__(self, calib_data_loader: DataLoader = None, num_calib_batches: int = 2, percentile: float = 99.99):
        super().__init__(phase=Phase.PRE_TRAINING)
        self.calib_data_loader = calib_data_loader
        self.num_calib_batches = num_calib_batches
        self.percentile = percentile

    def __call__(self, context: PhaseContext):
        self.calib_data_loader = self.calib_data_loader or context.valid_loader
        calibrate_model(model=context.net,
                        calib_data_loader=self.calib_data_loader,
                        method=context.quant_modules_calib_method,
                        num_calib_batches=self.num_calib_batches,
                        percentile=self.percentile)

        method_desc = context.quant_modules_calib_method + '_' + str(
            self.percentile) if context.quant_modules_calib_method == 'percentile' else context.quant_modules_calib_method
        if not context.ddp_silent_mode:
            context.sg_logger.add_checkpoint(tag='ckpt_calibrated_' + method_desc + '.pth', state_dict={"net": context.net})


class PostQATConversionCallback(PhaseCallback):
    def __init__(self, dummy_input_size):
        super().__init__(phase=Phase.POST_TRAINING)
        self.dummy_input_size = dummy_input_size

    def __call__(self, context: PhaseContext):
        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            if local_rank == 0:
                best_ckpt_path = os.path.join(context.checkpoints_dir_path, "ckpt_best.pth")
                onnx_path = os.path.join(context.checkpoints_dir_path, "ckpt_best.onnx")
                trt_path = os.path.join(context.checkpoints_dir_path, "ckpt_best.engine")

                load_checkpoint_to_model(ckpt_local_path=best_ckpt_path,
                                         net=context.net,
                                         load_weights_only=True,
                                         load_ema_as_net=context.training_params.ema,
                                         strict=True,
                                         load_backbone=False
                                         )

                export_qat_onnx(context.net, onnx_path, self.dummy_input_size)
                save_trt_engine_from_onnx_ckpt(onnx_path, trt_path)

                context.sg_logger.add_file("ckpt_best.onnx")
                context.sg_logger.add_file("ckpt_best.engine")
