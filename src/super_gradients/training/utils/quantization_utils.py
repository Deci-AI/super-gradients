from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.callbacks import Phase, PhaseCallback, PhaseContext

logger = get_logger(__name__)

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


def export_qat_onnx(model, onnx_filename, input_shape):
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


def calibrate_model(model, calib_data_loader, method="percentile", num_calib_batches=2, percentile=99.99):
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure
    elif method in ["percentile", "mse", "entropy", "max"]:
        with torch.no_grad():
            collect_stats(model, calib_data_loader, num_batches=num_calib_batches)

            # FOR PERCENTILE WE MUST PASS PERCENTILE VALUE THROUGH KWARGS,
            # SO IT WOULD BE PASSED TO module.load_calib_amax(**kwargs), AND IN OTHER METHODS WE MUST NOT PASS IT.
            if method == "precentile":
                compute_amax(model, method="percentile", percentile=percentile)
            else:
                compute_amax(model, method=method)
    else:
        raise ValueError(
            "Unsupported quantization calibration method, expected one of: percentile, mse, entropy, max, got " + str(
                method) + ".")


def collect_stats(model, data_loader, num_batches):
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


def compute_amax(model, **kwargs):
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
        context.sg_logger.add_checkpoint(tag='ckpt_calibrated_' + method_desc + '.pth', state_dict={"net": context.net})

class PostQATConversionCallback(PhaseCallback):

