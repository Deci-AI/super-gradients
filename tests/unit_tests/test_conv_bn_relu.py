import torch
import unittest
import torch.nn as nn
from super_gradients.modules import ConvBNReLU


class TestConvBnRelu(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = torch.randn(2, 32, 64, 64)
        self.test_kernels = [1, 3, 5]
        self.test_strides = [1, 2]
        self.use_activation = [True, False]
        self.use_normalization = [True, False]
        self.biases = [True, False]

    def test_conv_bn_relu(self):
        for use_normalization in self.use_normalization:
            for use_activation in self.use_activation:
                for kernel in self.test_kernels:
                    for stride in self.test_strides:
                        for bias in self.biases:

                            conv_bn_relu = ConvBNReLU(
                                32,
                                32,
                                kernel_size=kernel,
                                stride=stride,
                                padding=kernel // 2,
                                bias=bias,
                                use_activation=use_activation,
                                use_normalization=use_normalization,
                            )
                            conv_bn_relu_seq = nn.Sequential(
                                nn.Conv2d(32, 32, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=bias),
                                nn.BatchNorm2d(32) if use_normalization else nn.Identity(),
                                nn.ReLU() if use_activation else nn.Identity(),
                            )
                            # apply same conv weights and biases to compare output,
                            # because conv weight and biases have random initialization.
                            conv_bn_relu.seq[0].weight = conv_bn_relu_seq[0].weight
                            if bias:
                                conv_bn_relu.seq[0].bias = conv_bn_relu_seq[0].bias

                            self.assertTrue(
                                torch.equal(conv_bn_relu(self.sample), conv_bn_relu_seq(self.sample)),
                                msg=f"ConvBnRelu test failed for configuration: activation: "
                                f"{use_activation}, normalization: {use_normalization}, "
                                f"kernel: {kernel}, stride: {stride}",
                            )

    def test_conv_bn_relu_with_default_torch_arguments(self):
        """
        This test check that the default arguments behavior of ConvBNRelu module is aligned with torch modules defaults.
        Check that behavior of ConvBNRelu doesn't change with torch package upgrades.
        """
        conv_bn_relu = ConvBNReLU(32, 32, kernel_size=1)
        conv_bn_relu_defaults_torch = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU())
        # apply same conv weights and biases to compare output,
        # because conv weight and biases have random initialization.
        conv_bn_relu.seq[0].weight = conv_bn_relu_defaults_torch[0].weight
        conv_bn_relu.seq[0].bias = conv_bn_relu_defaults_torch[0].bias

        self.assertTrue(
            torch.equal(conv_bn_relu(self.sample), conv_bn_relu_defaults_torch(self.sample)),
            msg="ConvBnRelu test failed for defaults arguments configuration: ConvBNRelu default" "arguments are not aligned with torch defaults.",
        )


if __name__ == "__main__":
    unittest.main()
