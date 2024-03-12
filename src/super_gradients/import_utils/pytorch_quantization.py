import torch.nn.functional as F
import importlib

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
        import pip

        pip.main(["install", "pytorch_quantization==2.1.2", "--extra-index-url", "https://pypi.ngc.nvidia.com"])

        return import_pytorch_quantization_or_fail_with_instructions()
