import numpy as np
import torch
import torch.nn.functional as F
import importlib
from .install_utils import install_package

__all__ = ["import_pytorch_quantization_or_fail_with_instructions", "import_pytorch_quantization_or_install", "patch_pytorch_quantization_modules_if_needed"]


def __fixed_conv_transpose_2d_forward(self, input, output_size=None):
    from super_gradients.training.utils import torch_version_is_greater_or_equal

    if self.padding_mode != "zeros":
        raise ValueError("Only `zeros` padding mode is supported for QuantConvTranspose2d")

    if torch_version_is_greater_or_equal(1, 12):
        output_padding = self._output_padding(
            input=input,
            output_size=output_size,
            stride=self.stride,
            padding=self.padding,
            kernel_size=self.kernel_size,
            num_spatial_dims=2,
            dilation=self.dilation,
        )
    else:
        output_padding = self._output_padding(input=input, output_size=output_size, stride=self.stride, padding=self.padding, kernel_size=self.kernel_size)

    quant_input, quant_weight = self._quant(input)
    output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    return output


def __HistogramCalibrator_collect(self, x):
    """Collect histogram.
    This implementation is a copy of the original method from pytorch_quantization.calib.histogram.HistogramCalibrator
    which does computation in double-precision to prevent overflow.

    """
    from absl import logging

    if torch.min(x) < 0.0:
        logging.log_first_n(
            logging.INFO, ("Calibrator encountered negative values. It shouldn't happen after ReLU. " "Make sure this is the right tensor to calibrate."), 1
        )
        x = x.abs()

    x = x.float()

    if not self._torch_hist:
        x_np = x.cpu().detach().numpy()

        if self._skip_zeros:
            x_np = x_np[np.where(x_np != 0)]

        if self._calib_bin_edges is None and self._calib_hist is None:
            # first time it uses num_bins to compute histogram.
            self._calib_hist, self._calib_bin_edges = np.histogram(x_np, bins=self._num_bins)
            self._calib_hist = self._calib_hist.astype(np.int64)
        else:
            temp_amax = np.max(x_np)
            if temp_amax > self._calib_bin_edges[-1]:
                # increase the number of bins
                width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                # NOTE: np.arange may create an extra bin after the one containing temp_amax
                new_bin_edges = np.arange(self._calib_bin_edges[-1] + width, temp_amax + width, width)
                self._calib_bin_edges = np.hstack((self._calib_bin_edges, new_bin_edges))
            hist, self._calib_bin_edges = np.histogram(x_np, bins=self._calib_bin_edges)
            hist = hist.astype(np.int64)
            hist[: len(self._calib_hist)] += self._calib_hist
            self._calib_hist = hist
    else:
        # This branch of code is designed to match numpy version as close as possible
        with torch.no_grad():
            if self._skip_zeros:
                x = x[torch.where(x != 0)]

            # Because we collect histogram on absolute value, setting min=0 simplifying the rare case where
            # minimum value is not exactly 0 and first batch collected has larger min value than later batches
            x_max = x.max()
            if self._calib_bin_edges is None and self._calib_hist is None:
                self._calib_hist = torch.histc(x, bins=self._num_bins, min=0, max=x_max).double()
                self._calib_bin_edges = torch.linspace(0, x_max, self._num_bins + 1)
            else:
                if x_max > self._calib_bin_edges[-1]:
                    width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                    self._num_bins = int((x_max / width).ceil().item())
                    self._calib_bin_edges = torch.arange(0, x_max + width, width, device=x.device)

                hist = torch.histc(x, bins=self._num_bins, min=0, max=self._calib_bin_edges[-1]).double()
                hist[: self._calib_hist.numel()] += self._calib_hist
                self._calib_hist = hist


def __HistogramCalibrator_compute_amax(self, method: str, *, stride: int = 1, start_bin: int = 128, percentile: float = 99.99):
    """Compute the amax from the collected histogram.
    This implementation is a copy of the original method from pytorch_quantization.calib.histogram.HistogramCalibrator
    which does computation in long type to prevent overflow.

    Args:
        method: A string. One of ['entropy', 'mse', 'percentile']

    Keyword Arguments:
        stride: An integer. Default 1
        start_bin: An integer. Default 128
        percentils: A float number between [0, 100]. Default 99.99.

    Returns:
        amax: a tensor
    """
    from pytorch_quantization.calib.histogram import _compute_amax_entropy, _compute_amax_mse, _compute_amax_percentile

    if isinstance(self._calib_hist, torch.Tensor):
        calib_hist = self._calib_hist.long().cpu().numpy()
        calib_bin_edges = self._calib_bin_edges.cpu().numpy()
    else:
        calib_hist = self._calib_hist
        calib_bin_edges = self._calib_bin_edges

    if method == "entropy":
        calib_amax = _compute_amax_entropy(calib_hist, calib_bin_edges, self._num_bits, self._unsigned, stride, start_bin)
    elif method == "mse":
        calib_amax = _compute_amax_mse(calib_hist, calib_bin_edges, self._num_bits, self._unsigned, stride, start_bin)
    elif method == "percentile":
        calib_amax = _compute_amax_percentile(calib_hist, calib_bin_edges, percentile)
    else:
        raise TypeError("Unknown calibration method {}".format(method))

    return calib_amax.float()


def patch_pytorch_quantization_modules_if_needed():
    """
    This function change the forward() method of pytorch_quantization.nn.modules.quant_conv.QuantConvTranspose2d to
    support the change in the signature of _ConvTransposeNd._output_padding in torch 1.12.0
    It is a known issue in pytorch_quantization 2.1.2

    :return:
    """
    import pytorch_quantization.nn.modules.quant_conv
    from pytorch_quantization.version import __version__ as pytorch_quantization_version

    if pytorch_quantization_version == "2.1.2":
        # logger.debug("Patching pytorch_quantization modules")

        pytorch_quantization.nn.modules.quant_conv.QuantConvTranspose2d.forward = __fixed_conv_transpose_2d_forward
        pytorch_quantization.calib.histogram.HistogramCalibrator.collect = __HistogramCalibrator_collect
        pytorch_quantization.calib.histogram.HistogramCalibrator.compute_amax = __HistogramCalibrator_compute_amax


def import_pytorch_quantization_or_fail_with_instructions() -> None:
    package = "pytorch_quantization"
    try:
        importlib.import_module(package)
        globals()[package] = importlib.import_module(package)
        patch_pytorch_quantization_modules_if_needed()
    except ImportError:
        raise ImportError(
            "pytorch_quantization package is not installed. "
            "Please install it via `pip install pytorch_quantization==2.1.2 --extra-index-url https://pypi.ngc.nvidia.com`"
        )


def import_pytorch_quantization_or_install() -> None:
    package = "pytorch_quantization"

    try:
        importlib.import_module(package)

        patch_pytorch_quantization_modules_if_needed()
    except ImportError:
        install_package("pytorch_quantization==2.1.2", extra_index_url="https://pypi.ngc.nvidia.com")
        return import_pytorch_quantization_or_fail_with_instructions()
