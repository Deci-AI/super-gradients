from typing import Type, Union, Mapping, Any, Optional

import numpy as np
import torch
from torch import nn

from super_gradients.training.utils.regularization_utils import DropPath


class SuperRepBlock(nn.Module):
    """
    Repvgg block consists of three branches
    3x3: a branch of a 3x3 Convolution + BatchNorm + Activation
    1x1: a branch of a 1x1 Convolution + BatchNorm + Activation
    no_conv_branch: a branch with only BatchNorm which will only be used if
        input channel == output channel and use_residual_connection is True
    (usually in all but the first block of each stage)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        branches: int = 3,
        activation_type: Type[nn.Module] = nn.ReLU,
        activation_kwargs: Union[Mapping[str, Any], None] = None,
        se_type: Type[nn.Module] = nn.Identity,
        se_kwargs: Union[Mapping[str, Any], None] = None,
        build_residual_branches: bool = True,
        use_residual_connection: bool = True,
        use_alpha: bool = False,
        drop_path_probability: float = 0.0,
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
        :param build_residual_branches: Whether to initialize block with already fused paramters (for deployment)
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
        self.branches = branches

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = se_type(**se_kwargs)

        self.drop_path = DropPath(drop_path_probability) if drop_path_probability > 0.0 and branches > 1 else nn.Identity()

        if use_residual_connection and out_channels == in_channels and stride == 1:
            self.no_conv_branch = nn.BatchNorm2d(num_features=in_channels)
        else:
            self.no_conv_branch = None

        self.branches_3x3 = nn.ModuleList(
            [
                self._conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    groups=groups,
                )
                for _ in range(branches)
            ]
        )

        self.branch_weights = torch.nn.Parameter(torch.ones(self.branches), requires_grad=True)

        self.branch_1x1 = self._conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)

        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1

        if not build_residual_branches:
            self.fuse_block_residual_branches()
        else:
            self.build_residual_branches = True

    def forward(self, inputs):
        if not self.build_residual_branches:
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.no_conv_branch is None:
            id_out = 0
        else:
            id_out = self.no_conv_branch(inputs)

        weighted_sum = self.alpha * self.branch_1x1(inputs) + id_out
        for w, conv in zip(self.branch_weights, self.branches_3x3):
            weighted_sum += self.drop_path(w * conv(inputs))

        return self.nonlinearity(self.se(weighted_sum))

    def _get_equivalent_kernel_bias(self):
        """
        Fuses the 3x3, 1x1 and identity branches into a single 3x3 conv layer
        """
        # TODO - validate this implementation is correct
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.branch_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.no_conv_branch)

        kernel = self.alpha * self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        bias = self.alpha * bias1x1 + biasid
        for w, conv in zip(self.branch_weights, self.branches_3x3):
            kernel3x3, bias3x3 = self._fuse_bn_tensor(conv)
            kernel += kernel3x3 * w
            bias += bias3x3 * w

        return kernel, bias

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

    def _fuse_bn_tensor(self, branch):
        """
        Fusing of the batchnorm into the conv layer.
        If the branch is the identity branch (no conv) the kernel will simply be eye.
        :param branch:
        :type branch:
        :return:
        :rtype:
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_block_residual_branches(self):
        """
        converts a repvgg block from training model (with branches) to deployment mode (vgg like model)
        :return:
        :rtype:
        """
        if hasattr(self, "build_residual_branches") and not self.build_residual_branches:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.branches_3x3[0].conv.in_channels,
            out_channels=self.branches_3x3[0].conv.out_channels,
            kernel_size=self.branches_3x3[0].conv.kernel_size,
            stride=self.branches_3x3[0].conv.stride,
            padding=self.branches_3x3[0].conv.padding,
            dilation=self.branches_3x3[0].conv.dilation,
            groups=self.branches_3x3[0].conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("branches_3x3")
        self.__delattr__("branch_1x1")
        if hasattr(self, "no_conv_branch"):
            self.__delattr__("no_conv_branch")
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

    def prep_model_for_conversion(self, input_size: Optional[Union[tuple, list]] = None, **kwargs):
        self.fuse_block_residual_branches()


def fuse_repvgg_blocks_residual_branches(model: nn.Module):
    """
    Call fuse_block_residual_branches for all repvgg blocks in the model
    :param model: torch.nn.Module with repvgg blocks. Doesn't have to be entirely consists of repvgg.
    :type model: torch.nn.Module
    """
    if model.training:
        raise RuntimeError("To fuse RepVGG block residual branches, model must be on eval mode")
    from super_gradients.training.utils.utils import infer_model_device

    device = infer_model_device(model)
    for module in model.modules():
        if hasattr(module, "fuse_block_residual_branches"):
            module.fuse_block_residual_branches()
    model.build_residual_branches = False
    model.to(device)
