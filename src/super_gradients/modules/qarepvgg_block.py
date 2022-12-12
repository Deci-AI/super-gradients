from typing import Type, Union, Mapping, Any

import torch
from torch import nn
from super_gradients.modules.skip_connections import Residual


class QARepVGGBlock(nn.Module):
    """
    QARepVGG (S4) block from 'Make RepVGG Greater Again: A Quantization-aware Approach' (https://arxiv.org/pdf/2212.01593.pdf)
    It consists of three branches:

    3x3: a branch of a 3x3 Convolution + BatchNorm
    1x1: a branch of a 1x1 Convolution with bias
    identity: a Residual branch which will only be used if input channel == output channel and use_residual_connection is True
        (usually in all but the first block of each stage)

    BatchNorm is applied after summation of all three branches.
    In contrast to our implementation of RepVGGBlock, SE is applied AFTER NONLINEARITY in order to fuse Conv+Act in inference frameworks.

    This module converts to Conv+Act in a PTQ-friendly way by calling QARepVGGBlock.fuse_block_residual_branches().
    Has the same API as RepVGGBlock and is designed to be a plug-and-play replacement but is not compatible parameter-wise.
    Has less trainable parameters than RepVGGBlock because it has only 2 BatchNorms instead of 3.


                        |
                        |
        |---------------|---------------|
        |               |               |
       3x3             1x1              |
        |               |               |
    BatchNorm         +bias             |
        |               |               |
        |             *alpha            |
        |               |               |
        |---------------+---------------|
                        |
                    BatchNorm
                        |
                       Act
                        |
                       SE
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation_type: Type[nn.Module] = nn.ReLU,
        activation_kwargs: Union[Mapping[str, Any], None] = None,
        se_type: Type[nn.Module] = nn.Identity,
        se_kwargs: Union[Mapping[str, Any], None] = None,
        build_residual_branches: bool = True,
        use_residual_connection: bool = True,
        use_alpha: bool = False,
    ):
        """

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param activation_type: Type of the nonlinearity
        :param se_type: Type of the se block (Use nn.Identity to disable SE)
        :param stride: Output stride
        :param dilation: Dilation factor for 3x3 conv
        :param groups: Number of groups used in convolutions
        :param activation_kwargs: Additional arguments for instantiating activation module.
        :param se_kwargs: Additional arguments for instantiating SE module.
        :param build_residual_branches: Whether to initialize block with already fused parameters (for deployment)
        :param use_residual_connection: Whether to add input x to the output (Enabled in RepVGG, disabled in PP-Yolo)
        :param use_alpha: If True, enables additional learnable weighting parameter for 1x1 branch (PP-Yolo-E Plus)
        """
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {}
        if se_kwargs is None:
            se_kwargs = {}

        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.activation_type = activation_type
        self.activation_kwargs = activation_kwargs
        self.se_type = se_type
        self.se_kwargs = se_kwargs
        self.build_residual_branches = build_residual_branches
        self.use_residual_connection = use_residual_connection
        self.use_alpha = use_alpha

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = se_type(**se_kwargs)

        self.branch_3x3 = self._conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
        )
        self.branch_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=groups,
            bias=True,  # authors don't mention it specifically, it seems that it should be here
        )

        if use_residual_connection and out_channels == in_channels and stride == 1:
            self.identity = Residual()
            input_dim = self.in_channels // self.groups
            self.id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=torch.float32, device=self.branch_1x1.weight.device)
            for i in range(self.in_channels):
                self.id_tensor[i, i % input_dim, 1, 1] = 1
        else:
            self.identity = None

        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1.0

        self.post_bn = nn.BatchNorm2d(num_features=out_channels)

        if not build_residual_branches:
            self.fuse_block_residual_branches()

    def forward(self, inputs):
        if not self.build_residual_branches:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.identity is None:
            id_out = 0.0
        else:
            id_out = self.identity(inputs)

        return self.se(self.nonlinearity(self.post_bn(self.branch_3x3(inputs) + self.alpha * self.branch_1x1(inputs) + id_out)))

    def _get_equivalent_kernel_bias(self):
        """
        Fuses the 3x3, 1x1 and identity branches into a single 3x3 conv layer
        """
        kernel3x3, bias3x3 = self._fuse_branch(self.branch_3x3)  # legit fusion
        kernel1x1, bias1x1 = self._fuse_branch(self.branch_1x1)  # only extract weight and bias from Conv2d
        kernelid, biasid = self._fuse_branch(self.identity)  # get id_tensor and 0

        eq_kernel_3x3 = kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        eq_bias_3x3 = bias3x3 + self.alpha * bias1x1 + biasid

        return self._fuse_bn_tensor(
            eq_kernel_3x3,
            eq_bias_3x3,
            self.post_bn.running_mean,
            self.post_bn.running_var,
            self.post_bn.weight,
            self.post_bn.bias,
            self.post_bn.eps,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """
        padding the 1x1 convolution weights with zeros to be able to fuse the 3x3 conv layer with the 1x1
        :param kernel1x1: weights of the 1x1 convolution
        :type kernel1x1:
        :return: padded 1x1 weights
        :rtype:
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, kernel, bias, running_mean, running_var, gamma, beta, eps):
        std = torch.sqrt(running_var + eps)
        b = beta - gamma * running_mean / std
        A = gamma / std
        bias *= A
        A = A.expand_as(kernel.transpose(0, -1)).transpose(0, -1)

        return kernel * A, bias + b

    def _fuse_branch(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):  # our BN(3x3) branch
            return self._fuse_bn_tensor(
                branch.conv.weight,
                0,
                branch.bn.running_mean,
                branch.bn.running_var,
                branch.bn.weight,
                branch.bn.bias,
                branch.bn.eps,
            )

        if isinstance(branch, nn.Conv2d):  # our 1x1 branch
            if branch.bias is not None:
                return branch.weight, branch.bias
            else:
                return branch.weight, 0

        if isinstance(branch, Residual):  # our identity branch
            return self.id_tensor, 0

        raise ValueError("Unknown branch")

    def fuse_block_residual_branches(self):
        """
        converts a qarepvgg block from training model (with branches) to deployment mode (vgg like model)
        :return:
        :rtype:
        """
        if hasattr(self, "build_residual_branches") and not self.build_residual_branches:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.branch_3x3.conv.in_channels,
            out_channels=self.branch_3x3.conv.out_channels,
            kernel_size=self.branch_3x3.conv.kernel_size,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("branch_3x3")
        self.__delattr__("branch_1x1")
        if hasattr(self, "identity"):
            self.__delattr__("identity")
        if hasattr(self, "alpha"):
            self.__delattr__("alpha")
        self.build_residual_branches = False

    @staticmethod
    def _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, dilation=1):
        result = nn.Sequential()
        result.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                dilation=dilation,
            ),
        )
        result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        return result


if __name__ == "__main__":
    random_input = torch.randn([32, 3, 64, 64], dtype=torch.float32)

    block = QARepVGGBlock(3, 3)

    block.train()

    # collect BN statistics
    for i in range(1000):
        block(torch.randn([32, 3, 64, 64], dtype=torch.float32))

    block.eval()

    x_before = block(random_input)

    block.fuse_block_residual_branches()

    x_after = block(random_input)

    print((x_before - x_after).sum())  # original RepVGG block has 0.001-0.03 with this use case
