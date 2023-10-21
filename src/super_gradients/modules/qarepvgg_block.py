from typing import Type, Union, Mapping, Any, Optional

import torch
from torch import nn

from super_gradients.modules import RepVGGBlock
from super_gradients.modules.skip_connections import Residual


class QARepVGGBlock(nn.Module):
    """
    QARepVGG (S3/S4) block from 'Make RepVGG Greater Again: A Quantization-aware Approach' (https://arxiv.org/pdf/2212.01593.pdf)
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
        use_post_bn: bool = True,
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param activation_type: Type of the nonlinearity (nn.ReLU by default)
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
        :param use_post_bn: If True, adds BatchNorm after the sum of three branches (S4), if False, BatchNorm is not added (S3)
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
        self.use_post_bn = use_post_bn

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

        if use_residual_connection:
            assert out_channels == in_channels and stride == 1

            self.identity = Residual()

            input_dim = self.in_channels // self.groups
            id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3))
            for i in range(self.in_channels):
                id_tensor[i, i % input_dim, 1, 1] = 1.0

            self.id_tensor: Optional[torch.Tensor]
            self.register_buffer(
                name="id_tensor",
                tensor=id_tensor.to(dtype=self.branch_1x1.weight.dtype, device=self.branch_1x1.weight.device),
                persistent=False,  # so it's not saved in state_dict
            )
        else:
            self.identity = None

        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1.0

        if self.use_post_bn:
            self.post_bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.post_bn = nn.Identity()

        # placeholder to correctly register parameters
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

        self.partially_fused = False
        self.fully_fused = False

        if not build_residual_branches:
            self.fuse_block_residual_branches()

    def forward(self, inputs):
        if self.fully_fused:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.partially_fused:
            return self.se(self.nonlinearity(self.post_bn(self.rbr_reparam(inputs))))

        if self.identity is None:
            id_out = 0.0
        else:
            id_out = self.identity(inputs)

        x_3x3 = self.branch_3x3(inputs)
        x_1x1 = self.alpha * self.branch_1x1(inputs)

        branches = x_3x3 + x_1x1 + id_out

        out = self.nonlinearity(self.post_bn(branches))
        se = self.se(out)

        return se

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

        fused_kernel = kernel * A_
        fused_bias = bias * A + b

        return fused_kernel, fused_bias

    def full_fusion(self):
        """Fuse everything into Conv-Act-SE, non-trainable, parameters detached
        converts a qarepvgg block from training model (with branches) to deployment mode (vgg like model)
        :return:
        :rtype:
        """
        if self.fully_fused:
            return

        if not self.partially_fused:
            self.partial_fusion()

        if self.use_post_bn:
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

        self.partially_fused = False
        self.fully_fused = True

    def partial_fusion(self):
        """Fuse branches into a single kernel, leave post_bn unfused, leave parameters differentiable"""
        if self.partially_fused:
            return

        if self.fully_fused:
            # TODO: we actually can, all we need to do is insert the properly initialized post_bn back
            # init is not trivial, so not implemented for now
            raise NotImplementedError("QARepVGGBlock can't be converted to partially fused from fully fused")

        kernel, bias = self._get_equivalent_kernel_bias_for_branches()

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

        self.partially_fused = True
        self.fully_fused = False

    def fuse_block_residual_branches(self):
        # inference frameworks will take care of resulting conv-bn-act-se
        # no need to fuse post_bn prematurely if it is there
        # call self.full_fusion() if you need it
        self.partial_fusion()

    def from_repvgg(self, src: RepVGGBlock):
        raise NotImplementedError

    def prep_model_for_conversion(self, input_size: Optional[Union[tuple, list]] = None, full_fusion: bool = True, **kwargs):
        """Prepare the QARepVGGBlock for conversion.

        :WARNING: the default `full_fusion=True` will make the block non-trainable.

        :param full_fusion: If True, performs full fusion, converting the block into a non-trainable, fully fused block.
                            If False, performs partial fusion, slower for inference but still trainable.
        """

        if full_fusion:
            self.full_fusion()
        else:
            self.partial_fusion()
