'''
Repvgg Pytorch Implementation. This model trains a vgg with residual blocks
but during inference (in deployment mode) will convert the model to vgg model.
Pretrained models: https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq
Refrerences:
    [1] https://github.com/DingXiaoH/RepVGG
    [2] https://arxiv.org/pdf/2101.03697.pdf

Based on https://github.com/DingXiaoH/RepVGG
'''
from typing import Union

import torch.nn as nn
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from super_gradients.training.models import SgModule
import torch.nn.functional as F
from super_gradients.training.utils.module_utils import fuse_repvgg_blocks_residual_branches
from super_gradients.training.utils.utils import get_param


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, dilation=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False, dilation=dilation))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    '''
    Repvgg block consists of three branches
    3x3: a branch of a 3x3 convolution + batchnorm + relu
    1x1: a branch of a 1x1 convolution + batchnorm + relu
    no_conv_branch: a branch with only batchnorm which will only be used if input channel == output channel
    (usually in all but the first block of each stage)
    '''
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, build_residual_branches=True, use_relu=True,
                 use_se=False):

        super(RepVGGBlock, self).__init__()

        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == dilation

        self.nonlinearity = nn.ReLU() if use_relu else nn.Identity()
        self.se = nn.Identity() if not use_se else SEBlock(out_channels, internal_neurons=out_channels // 16)

        self.no_conv_branch = nn.BatchNorm2d(
            num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.branch_3x3 = conv_bn(in_channels=in_channels, out_channels=out_channels, dilation=dilation,
                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.branch_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                  padding=0, groups=groups)

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

        return self.nonlinearity(self.se(self.branch_3x3(inputs) + self.branch_1x1(inputs) + id_out))

    def _get_equivalent_kernel_bias(self):
        """
        Fuses the 3x3, 1x1 and identity branches into a single 3x3 conv layer
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.branch_3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.branch_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.no_conv_branch)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

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
            if not hasattr(self, 'id_tensor'):
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
        self.rbr_reparam = nn.Conv2d(in_channels=self.branch_3x3.conv.in_channels, out_channels=self.branch_3x3.conv.out_channels,
                                     kernel_size=self.branch_3x3.conv.kernel_size, stride=self.branch_3x3.conv.stride,
                                     padding=self.branch_3x3.conv.padding, dilation=self.branch_3x3.conv.dilation, groups=self.branch_3x3.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('branch_3x3')
        self.__delattr__('branch_1x1')
        if hasattr(self, 'no_conv_branch'):
            self.__delattr__('no_conv_branch')
        self.build_residual_branches = False


class RepVGG(SgModule):

    def __init__(self, struct, num_classes=1000, width_multiplier=None,
                 build_residual_branches=True, use_se=False, backbone_mode=False, in_channels=3):
        """
        :param struct: list containing number of blocks per repvgg stage
        :param num_classes: number of classes if nut in backbone mode
        :param width_multiplier: list of per stage width multiplier or float if using single value for all stages
        :param build_residual_branches: whether to add residual connections or not
        :param use_se: use squeeze and excitation layers
        :param backbone_mode: if true, dropping the final linear layer
        :param in_channels: input channels
        """
        super(RepVGG, self).__init__()

        if isinstance(width_multiplier, float):
            width_multiplier = [width_multiplier] * 4
        else:
            assert len(width_multiplier) == 4

        self.build_residual_branches = build_residual_branches
        self.use_se = use_se
        self.backbone_mode = backbone_mode

        self.in_planes = int(64 * width_multiplier[0])

        self.stem = RepVGGBlock(in_channels=in_channels, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                build_residual_branches=build_residual_branches, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), struct[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), struct[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), struct[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), struct[3], stride=2)
        if not self.backbone_mode:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

        if not build_residual_branches:
            self.eval()  # fusing has to be made in eval mode. When called in init, model will be built in eval mode
            fuse_repvgg_blocks_residual_branches(self)

        self.final_width_mult = width_multiplier[3]

    def _make_stage(self, planes, struct, stride):
        strides = [stride] + [1] * (struct - 1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=1, build_residual_branches=self.build_residual_branches,
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        if not self.backbone_mode:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        if self.build_residual_branches:
            fuse_repvgg_blocks_residual_branches(self)

    def train(self, mode: bool = True):

        assert not mode or self.build_residual_branches, "Trying to train a model without residual branches, " \
                                                         "set arch_params.build_residual_branches to True and retrain the model"
        super(RepVGG, self).train(mode=mode)

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.linear = new_head
        else:
            self.linear = nn.Linear(int(512 * self.final_width_mult), new_num_classes)


class RepVggCustom(RepVGG):
    def __init__(self, arch_params):
        super().__init__(struct=arch_params.struct, num_classes=arch_params.num_classes,
                         width_multiplier=arch_params.width_multiplier,
                         build_residual_branches=arch_params.build_residual_branches,
                         use_se=get_param(arch_params, 'use_se', False),
                         backbone_mode=get_param(arch_params, 'backbone_mode', False),
                         in_channels=get_param(arch_params, 'in_channels', 3))


class RepVggA0(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5])
        super().__init__(arch_params=arch_params)


class RepVggA1(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5])
        super().__init__(arch_params=arch_params)


class RepVggA2(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75])
        super().__init__(arch_params=arch_params)


class RepVggB0(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[4, 6, 16, 1], width_multiplier=[1, 1, 1, 2.5])
        super().__init__(arch_params=arch_params)


class RepVggB1(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4])
        super().__init__(arch_params=arch_params)


class RepVggB2(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5])
        super().__init__(arch_params=arch_params)


class RepVggB3(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5])
        super().__init__(arch_params=arch_params)


class RepVggD2SE(RepVggCustom):
    def __init__(self, arch_params):
        arch_params.override(struct=[8, 14, 24, 1], width_multiplier=[2.5, 2.5, 2.5, 5])
        super().__init__(arch_params=arch_params)
