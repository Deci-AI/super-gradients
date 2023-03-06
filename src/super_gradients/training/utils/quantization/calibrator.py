"""
Quantization utilities

Methods are based on:
https://github.com/NVIDIA/TensorRT/blob/51a4297753d3e12d0eed864be52400f429a6a94c/tools/pytorch-quantization/examples/torchvision/classification_flow.py#L385

(Licensed under the Apache License, Version 2.0)
"""
import logging

import torch
from tqdm import tqdm

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.distributed_training_utils import get_local_rank, get_world_size
from torch.distributed import all_gather

logger = get_logger(__name__)

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


class QuantizationCalibrator:
    def __init__(self, torch_hist: bool = True, verbose: bool = True) -> None:
        if _imported_pytorch_quantization_failure is not None:
            raise _imported_pytorch_quantization_failure
        super().__init__()
        self.verbose = verbose
        self.torch_hist = torch_hist

    def calibrate_model(
        self,
        model: torch.nn.Module,
        calib_data_loader: torch.utils.data.DataLoader,
        method: str = "percentile",
        num_calib_batches: int = 2,
        percentile: float = 99.99,
    ):
        """
        Calibrates torch model with quantized modules.

        :param model:               torch.nn.Module, model to perfrom the calibration on.
        :param calib_data_loader:   torch.utils.data.DataLoader, data loader of the calibration dataset.
                                    Assumes that the first element of the tuple is the input image.
        :param method:              str, One of [percentile, mse, entropy, max].
                                    Statistics method for amax computation of the quantized modules
                                    (Default=percentile).
        :param num_calib_batches:   int, number of batches to collect the statistics from.
        :param percentile:          float, percentile value to use when SgModel,quant_modules_calib_method='percentile'.
                                    Discarded when other methods are used (Default=99.99).

        """

        logging_level = logging.getLogger("absl").getEffectiveLevel()
        if not self.verbose:  # suppress pytorch-quantization spam
            logging.getLogger("absl").setLevel("ERROR")

        acceptable_methods = ["percentile", "mse", "entropy", "max"]
        if method in acceptable_methods:
            with torch.no_grad():
                device = next(model.parameters()).device

                self._collect_stats(model, calib_data_loader, num_batches=num_calib_batches)
                # FOR PERCENTILE WE MUST PASS PERCENTILE VALUE THROUGH KWARGS,
                # SO IT WOULD BE PASSED TO module.load_calib_amax(**kwargs), AND IN OTHER METHODS WE MUST NOT PASS IT.
                if method == "precentile":
                    self._compute_amax(model, method="percentile", percentile=percentile)
                else:
                    self._compute_amax(model, method=method)

                model.to(device)
        else:
            raise ValueError(f"Unsupported quantization calibration method, " f"expected one of: {'.'.join(acceptable_methods)}, however, received: {method}")

        logging.getLogger("absl").setLevel(logging_level)

    def _collect_stats(self, model, data_loader, num_batches):
        """Feed data to the network and collect statistics"""
        local_rank = get_local_rank()
        world_size = get_world_size()

        device = next(model.parameters()).device

        # Enable calibrators
        self._enable_calibrators(model)

        # Feed data to the network for collecting stats
        for i, (image, *_) in tqdm(enumerate(data_loader), total=num_batches, disable=local_rank > 0):
            if world_size > 1:
                all_batches = [torch.zeros_like(image, device=device) for _ in range(world_size)]
                all_gather(all_batches, image.to(device=device))
            else:
                all_batches = [image]

            for local_image in all_batches:
                model(local_image.to(device=device))
            if i >= num_batches:
                break

        # Disable calibrators
        self._disable_calibrators(model)

    def _disable_calibrators(self, model):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_calib()
                    module.enable_quant()
                else:
                    module.enable()

    def reset_calibrators(self, model):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module._calibrator.reset()  # release memory

    def _enable_calibrators(self, model):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.HistogramCalibrator):
                        module._calibrator._torch_hist = self.torch_hist  # TensorQuantizer does not expose it as API
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

    def _compute_amax(self, model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

                if hasattr(module, "clip"):
                    module.init_learn_amax()

                if self.verbose:
                    print(f"{name:40}: {module}")
