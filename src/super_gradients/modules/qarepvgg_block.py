from typing import Type, Union, Mapping, Any

import torch
from torch import nn

from super_gradients.modules import RepVGGBlock
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
        use_1x1_bias: bool = True,
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
        :param use_1x1_bias: If True, enables bias in the 1x1 convolution, authors don't mention it specifically
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
        self.use_residual_connection = use_residual_connection
        self.use_alpha = use_alpha
        self.use_1x1_bias = use_1x1_bias

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = se_type(**se_kwargs)

        self.branch_3x3 = nn.Sequential()
        self.branch_3x3.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation,
            ),
        )
        self.branch_3x3.add_module("bn", nn.BatchNorm2d(num_features=out_channels))

        self.branch_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=groups,
            bias=use_1x1_bias,
        )

        if use_residual_connection and out_channels == in_channels and stride == 1:
            self.identity = Residual()

            input_dim = self.in_channels // self.groups
            self.id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3))
            for i in range(self.in_channels):
                self.id_tensor[i, i % input_dim, 1, 1] = 1.0
            self.id_tensor = self.id_tensor.to(dtype=self.branch_1x1.weight.dtype, device=self.branch_1x1.weight.device)
        else:
            self.identity = None

        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1.0

        self.post_bn = nn.BatchNorm2d(num_features=out_channels)

        self.qat_mode = False
        self.deploy_mode = False

        if not build_residual_branches:
            self.fuse_block_residual_branches()

    def forward(self, inputs):
        if self.deploy_mode:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.qat_mode:
            return self.se(self.nonlinearity(self.post_bn(self.rbr_reparam(inputs))))

        if self.identity is None:
            id_out = 0.0
        else:
            id_out = self.identity(inputs)

        return self.se(self.nonlinearity(self.post_bn(self.branch_3x3(inputs) + self.alpha * self.branch_1x1(inputs) + id_out)))

    def _get_equivalent_kernel_bias_for_branches(self):
        """
        Fuses the 3x3, 1x1 and identity branches into a single 3x3 conv layer
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(
            self.branch_3x3.conv.weight,
            0,
            self.branch_3x3.bn.running_mean,
            self.branch_3x3.bn.running_var,
            self.branch_3x3.bn.weight,
            self.branch_3x3.bn.bias,
            self.branch_3x3.bn.eps,
        )

        kernel1x1 = self._pad_1x1_to_3x3_tensor(self.branch_1x1.weight)
        bias1x1 = self.branch_1x1.bias if self.branch_1x1.bias is not None else 0

        kernelid = self.id_tensor if self.identity is not None else 0
        biasid = 0

        eq_kernel_3x3 = kernel3x3 + self.alpha * kernel1x1 + kernelid
        eq_bias_3x3 = bias3x3 + self.alpha * bias1x1 + biasid

        return eq_kernel_3x3, eq_bias_3x3

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
        A_ = A.expand_as(kernel.transpose(0, -1)).transpose(0, -1)

        return kernel * A_, bias * A + b

    def prepare_for_deploy(self):
        """Fuse everything into Conv-Act-SE, non-trainable, parameters detached
        converts a qarepvgg block from training model (with branches) to deployment mode (vgg like model)
        :return:
        :rtype:
        """
        if self.deploy_mode:
            return

        if not self.qat_mode:
            self.prepare_for_qat()

        eq_kernel, eq_bias = self._fuse_bn_tensor(
            self.rbr_reparam.weight,
            self.rbr_reparam.bias,
            self.post_bn.running_mean,
            self.post_bn.running_var,
            self.post_bn.weight,
            self.post_bn.bias,
            self.post_bn.eps,
        )

        self.rbr_reparam.weight.data = eq_kernel
        self.rbr_reparam.bias.data = eq_bias

        for para in self.parameters():
            para.detach_()

        if hasattr(self, "post_bn"):
            self.__delattr__("post_bn")

        self.qat_mode = False
        self.deploy_mode = True

    def prepare_for_qat(self):
        """Fuse branches into a single kernel, leave post_bn unfused, leave parameters differentiable"""
        if self.qat_mode:
            return

        if self.deploy_mode:
            # TODO: we actually can, all we need to do is insert the properly initialized post_bn back
            # init is not trivial, so not implemented for now
            raise NotImplementedError("QARepVGGBlock can't be converted to QAT mode from deploy mode")

        kernel, bias = self._get_equivalent_kernel_bias_for_branches()
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

        self.__delattr__("branch_3x3")
        self.__delattr__("branch_1x1")
        if hasattr(self, "identity"):
            self.__delattr__("identity")
        if hasattr(self, "alpha"):
            self.__delattr__("alpha")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

        self.qat_mode = True
        self.deploy_mode = False

    def fuse_block_residual_branches(self):
        self.prepare_for_deploy()

    def from_repvgg(self, repvgg_block: RepVGGBlock):
        # src.conv3x3 -> self.conv3x3
        # src.bn3x3 -> self.bn3x3

        # src.conv1x1+src.bn1x1 -> self.conv1x1

        # fuse(src.ncb) -> self.conv1x1
        # fuse conv1x1, bn1x1
        raise NotImplementedError


if __name__ == "__main__":
    random_input = torch.randn([32, 3, 64, 64], dtype=torch.float32)

    block = QARepVGGBlock(3, 3, use_1x1_bias=False)

    block.train()

    # collect BN statistics
    for _ in range(10):
        block(torch.randn([32, 3, 64, 64], dtype=torch.float32))

    block.eval()

    x_before = block(random_input)

    block.prepare_for_qat()
    x_after = block(random_input)
    print((x_before - x_after).abs().sum())  # original RepVGG block has 0.001-0.03 with this use case

    block.prepare_for_deploy()
    x_after = block(random_input)
    print((x_before - x_after).abs().sum())
