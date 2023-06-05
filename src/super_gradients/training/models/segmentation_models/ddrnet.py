import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.models.classification_models.resnet import BasicResNetBlock, Bottleneck
from super_gradients.training.models.segmentation_models.segmentation_module import SegmentationModule
from super_gradients.training.utils import get_param, HpmStruct

"""
paper: Deep Dual-resolution Networks for Real-time and
Accurate Semantic Segmentation of Road Scenes ( https://arxiv.org/pdf/2101.06085.pdf )
code from git repo: https://github.com/ydhongHIT/DDRNet
"""


def ConvBN(in_channels: int, out_channels: int, kernel_size: int, bias=True, stride=1, padding=0, add_relu=False):
    seq = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding), nn.BatchNorm2d(out_channels)]
    if add_relu:
        seq.append(nn.ReLU(inplace=True))
    return nn.Sequential(*seq)


def _make_layer(block, in_planes, planes, num_blocks, stride=1, expansion=1):
    layers = []
    layers.append(block(in_planes, planes, stride, final_relu=num_blocks > 1, expansion=expansion))
    in_planes = planes * expansion
    if num_blocks > 1:
        for i in range(1, num_blocks):
            if i == (num_blocks - 1):
                layers.append(block(in_planes, planes, stride=1, final_relu=False, expansion=expansion))
            else:
                layers.append(block(in_planes, planes, stride=1, final_relu=True, expansion=expansion))

    return nn.Sequential(*layers)


class DAPPMBranch(nn.Module):
    def __init__(self, kernel_size: int, stride: int, in_planes: int, branch_planes: int, inter_mode: str = "bilinear"):
        """
        A DAPPM branch
        :param kernel_size: the kernel size for the average pooling
                when stride=0 this parameter is omitted and AdaptiveAvgPool2d over all the input is performed
        :param stride: stride for the average pooling
                when stride=0: an AdaptiveAvgPool2d over all the input is performed (output is 1x1)
                when stride=1: no average pooling is performed
                when stride>1: average polling is performed (scaling the input down and up again)
        :param in_planes:
        :param branch_planes: width after the the first convolution
        :param inter_mode: interpolation mode for upscaling
        """

        super().__init__()
        down_list = []
        if stride == 0:
            # when stride is 0 average pool all the input to 1x1
            down_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif stride == 1:
            # when stride id 1 no average pooling is used
            pass
        else:
            down_list.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=stride))

        down_list.append(nn.BatchNorm2d(in_planes))
        down_list.append(nn.ReLU(inplace=True))
        down_list.append(nn.Conv2d(in_planes, branch_planes, kernel_size=1, bias=False))

        self.down_scale = nn.Sequential(*down_list)
        self.up_scale = UpscaleOnline(inter_mode)

        if stride != 1:
            self.process = nn.Sequential(
                nn.BatchNorm2d(branch_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
            )

    def forward(self, x):
        """
        All branches of the DAPPM but the first one receive the output of the previous branch as a second input
        :param x: in branch 0 - the original input of the DAPPM. in other branches - a list containing the original
        input and the output of the previous branch.
        """

        if isinstance(x, list):
            output_of_prev_branch = x[1]
            x = x[0]
        else:
            output_of_prev_branch = None

        in_width = x.shape[-1]
        in_height = x.shape[-2]
        out = self.down_scale(x)
        out = self.up_scale(out, output_height=in_height, output_width=in_width)

        if output_of_prev_branch is not None:
            out = self.process(out + output_of_prev_branch)

        return out


class DAPPM(nn.Module):
    def __init__(self, in_planes: int, branch_planes: int, out_planes: int, kernel_sizes: list, strides: list, inter_mode: str = "bilinear"):
        super().__init__()

        assert len(kernel_sizes) == len(strides), "len of kernel_sizes and strides must be the same"
        self.branches = nn.ModuleList()
        for kernel_size, stride in zip(kernel_sizes, strides):
            self.branches.append(DAPPMBranch(kernel_size=kernel_size, stride=stride, in_planes=in_planes, branch_planes=branch_planes, inter_mode=inter_mode))

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * len(self.branches)),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * len(self.branches), out_planes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x_list = []
        for i, branch in enumerate(self.branches):
            if i == 0:
                x_list.append(branch(x))
            else:
                x_list.append(branch([x, x_list[i - 1]]))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class SegmentHead(nn.Module):
    def __init__(self, in_planes: int, inter_planes: int, out_planes: int, scale_factor: int, inter_mode: str = "bilinear"):
        """
        Last stage of the segmentation network.
        Reduces the number of output planes (usually to num_classes) while increasing the size by scale_factor
        :param in_planes: width of input
        :param inter_planes: width of internal conv. must be a multiple of scale_factor^2 when inter_mode=pixel_shuffle
        :param out_planes: output width
        :param scale_factor: scaling factor
        :param inter_mode: one of nearest, linear, bilinear, bicubic, trilinear, area or pixel_shuffle.
        when set to pixel_shuffle, an nn.PixelShuffle will be used for scaling
        """
        super().__init__()

        if inter_mode == "pixel_shuffle":
            assert inter_planes % (scale_factor ^ 2) == 0, "when using pixel_shuffle, inter_planes must be a multiple of scale_factor^2"

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.relu = nn.ReLU(inplace=True)

        if inter_mode == "pixel_shuffle":
            self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=1, padding=0, bias=True)
            self.upscale = nn.PixelShuffle(scale_factor)
        else:
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=1, padding=0, bias=True)
            self.upscale = nn.Upsample(scale_factor=scale_factor, mode=inter_mode)

        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        out = self.upscale(out)

        return out


class UpscaleOnline(nn.Module):
    """
    In some cases the required scale/size for the scaling is known only when the input is received.
    This class support such cases. only the interpolation mode is set in advance.
    """

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, x, output_height: int, output_width: int):
        return F.interpolate(x, size=[output_height, output_width], mode=self.mode)


class DDRBackBoneBase(nn.Module):
    """A base class defining functions that must be supported by DDRBackBones"""

    def validate_backbone_attributes(self):
        expected_attributes = ["stem", "layer1", "layer2", "layer3", "layer4", "input_channels"]
        for attribute in expected_attributes:
            assert hasattr(self, attribute), f"Invalid backbone - attribute '{attribute}' is missing"

    def get_backbone_output_number_of_channels(self):
        """Return a dictionary of the shapes of each output of the backbone to determine the in_channels of the
        skip and compress layers"""
        output_shapes = {}
        x = torch.randn(1, self.input_channels, 320, 320)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        output_shapes["layer2"] = x.shape[1]
        for layer in self.layer3:
            x = layer(x)
        output_shapes["layer3"] = x.shape[1]
        x = self.layer4(x)
        output_shapes["layer4"] = x.shape[1]
        return output_shapes


class BasicDDRBackBone(DDRBackBoneBase):
    def __init__(self, block: nn.Module.__class__, width: int, layers: list, input_channels: int, layer3_repeats: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            ConvBN(in_channels=input_channels, out_channels=width, kernel_size=3, stride=2, padding=1, add_relu=True),
            ConvBN(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1, add_relu=True),
        )
        self.layer1 = _make_layer(block=block, in_planes=width, planes=width, num_blocks=layers[0])
        self.layer2 = _make_layer(block=block, in_planes=width, planes=width * 2, num_blocks=layers[1], stride=2)
        self.layer3 = nn.ModuleList(
            [_make_layer(block=block, in_planes=width * 2, planes=width * 4, num_blocks=layers[2], stride=2)]
            + [_make_layer(block=block, in_planes=width * 4, planes=width * 4, num_blocks=layers[2], stride=1) for _ in range(layer3_repeats - 1)]
        )
        self.layer4 = _make_layer(block=block, in_planes=width * 4, planes=width * 8, num_blocks=layers[3], stride=2)


class RegnetDDRBackBone(DDRBackBoneBase):
    """
    Translation of Regnet to fit DDR model
    """

    def __init__(self, regnet_module: nn.Module.__class__):
        super().__init__()
        self.input_channels = regnet_module.net.stem.conv.in_channels
        self.stem = regnet_module.net.stem
        self.layer1 = regnet_module.net.stage_0
        self.layer2 = regnet_module.net.stage_1
        self.layer3 = nn.ModuleList([regnet_module.net.stage_2])
        self.layer4 = regnet_module.net.stage_3


class DDRNet(SegmentationModule):
    def __init__(
        self,
        backbone: DDRBackBoneBase.__class__,
        additional_layers: list,
        upscale_module: nn.Module,
        num_classes: int,
        highres_planes: int,
        spp_width: int,
        head_width: int,
        use_aux_heads: bool = False,
        ssp_inter_mode: str = "bilinear",
        segmentation_inter_mode: str = "bilinear",
        skip_block: nn.Module.__class__ = None,
        layer5_block: nn.Module.__class__ = Bottleneck,
        layer5_bottleneck_expansion: int = 2,
        classification_mode=False,
        spp_kernel_sizes: list = [1, 5, 9, 17, 0],
        spp_strides: list = [1, 2, 4, 8, 0],
        layer3_repeats: int = 1,
    ):
        """

        :param backbone: the low resolution branch of DDR, expected to have specific attributes in the class
        :param additional_layers: list of num blocks for the highres stage and layer5
        :param upscale_module: upscale to use in the backbone (DAPPM and Segmentation head are using bilinear interpolation)
        :param num_classes: number of classes
        :param highres_planes: number of channels in the high resolution net
        :param use_aux_heads: add a second segmentation head (fed from after compress3 + upscale). this head can be used
        during training (see paper https://arxiv.org/pdf/2101.06085.pdf for details)
        :param ssp_inter_mode: the interpolation used in the SPP block
        :param segmentation_inter_mode: the interpolation used in the segmentation head
        :param skip_block: allows specifying a different block (from 'block') for the skip layer
        :param layer5_block: type of block to use in layer5 and layer5_skip
        :param layer5_bottleneck_expansion: determines the expansion rate for Bottleneck block
        :param spp_kernel_sizes: list of kernel sizes for the spp module pooling
        :param spp_strides: list of strides for the spp module pooling
        :param layer3_repeats: number of times to repeat the 3rd stage of ddr model, including the paths interchange
         modules.
        """

        super().__init__(use_aux_heads=use_aux_heads)
        self.use_aux_heads = use_aux_heads
        self.upscale = upscale_module
        self.ssp_inter_mode = ssp_inter_mode
        self.segmentation_inter_mode = segmentation_inter_mode
        self.relu = nn.ReLU(inplace=False)
        self.classification_mode = classification_mode
        self.layer3_repeats = layer3_repeats

        assert not (use_aux_heads and classification_mode), "auxiliary head cannot be used in classification mode"

        assert isinstance(backbone, DDRBackBoneBase), "The backbone must inherit from AbstractDDRBackBone"
        self._backbone = backbone
        self._backbone.validate_backbone_attributes()
        out_chan_backbone = self._backbone.get_backbone_output_number_of_channels()

        # Repeat r-times layer4
        self.compression3, self.down3, self.layer3_skip = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(layer3_repeats):
            self.compression3.append(ConvBN(in_channels=out_chan_backbone["layer3"], out_channels=highres_planes, kernel_size=1, bias=False))
            self.down3.append(ConvBN(in_channels=highres_planes, out_channels=out_chan_backbone["layer3"], kernel_size=3, stride=2, padding=1, bias=False))
            self.layer3_skip.append(
                _make_layer(
                    in_planes=out_chan_backbone["layer2"] if i == 0 else highres_planes,
                    planes=highres_planes,
                    block=skip_block,
                    num_blocks=additional_layers[1],
                )
            )

        self.compression4 = ConvBN(in_channels=out_chan_backbone["layer4"], out_channels=highres_planes, kernel_size=1, bias=False)

        self.down4 = nn.Sequential(
            ConvBN(in_channels=highres_planes, out_channels=highres_planes * 2, kernel_size=3, stride=2, padding=1, bias=False, add_relu=True),
            ConvBN(in_channels=highres_planes * 2, out_channels=out_chan_backbone["layer4"], kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.layer4_skip = _make_layer(block=skip_block, in_planes=highres_planes, planes=highres_planes, num_blocks=additional_layers[2])
        self.layer5_skip = _make_layer(
            block=layer5_block, in_planes=highres_planes, planes=highres_planes, num_blocks=additional_layers[3], expansion=layer5_bottleneck_expansion
        )

        # when training the backbones on Imagenet:
        #  - layer 5 has stride 1
        #  - a new high_to_low_fusion is added with to 3x3 convs with stride 2 (and double the width)
        #  - a classification head is placed instead of the segmentation head
        if self.classification_mode:
            self.layer5 = _make_layer(
                block=layer5_block,
                in_planes=out_chan_backbone["layer4"],
                planes=out_chan_backbone["layer4"],
                num_blocks=additional_layers[0],
                expansion=layer5_bottleneck_expansion,
            )

            highres_planes_out = highres_planes * layer5_bottleneck_expansion
            self.high_to_low_fusion = nn.Sequential(
                ConvBN(in_channels=highres_planes_out, out_channels=highres_planes_out * 2, kernel_size=3, stride=2, padding=1, add_relu=True),
                ConvBN(
                    in_channels=highres_planes_out * 2,
                    out_channels=out_chan_backbone["layer4"] * layer5_bottleneck_expansion,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    add_relu=True,
                ),
            )

            self.average_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(in_features=out_chan_backbone["layer4"] * layer5_bottleneck_expansion, out_features=num_classes)

        else:
            self.layer5 = _make_layer(
                block=layer5_block,
                in_planes=out_chan_backbone["layer4"],
                planes=out_chan_backbone["layer4"],
                num_blocks=additional_layers[0],
                stride=2,
                expansion=layer5_bottleneck_expansion,
            )

            self.spp = DAPPM(
                in_planes=out_chan_backbone["layer4"] * layer5_bottleneck_expansion,
                branch_planes=spp_width,
                out_planes=highres_planes * layer5_bottleneck_expansion,
                inter_mode=self.ssp_inter_mode,
                kernel_sizes=spp_kernel_sizes,
                strides=spp_strides,
            )

            self.final_layer = SegmentHead(highres_planes * layer5_bottleneck_expansion, head_width, num_classes, 8, inter_mode=self.segmentation_inter_mode)

            if self.use_aux_heads:
                self.seghead_extra = SegmentHead(highres_planes, head_width, num_classes, 8, inter_mode=self.segmentation_inter_mode)

        self.highres_planes = highres_planes
        self.layer5_bottleneck_expansion = layer5_bottleneck_expansion
        self.head_width = head_width
        self.init_params()

    @property
    def backbone(self):
        """
        Create a fake backbone module to load backbone pre-trained weights.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    ("_backbone", self._backbone),
                    ("compression3", self.compression3),
                    ("compression4", self.compression4),
                    ("down3", self.down3),
                    ("down4", self.down4),
                    ("layer3_skip", self.layer3_skip),
                    ("layer4_skip", self.layer4_skip),
                    ("layer4_skip", self.layer4_skip),
                    ("layer5_skip", self.layer5_skip),
                ]
            )
        )

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self._backbone.stem(x)
        x = self._backbone.layer1(x)
        x = self._backbone.layer2(self.relu(x))

        # Repeat layer 3
        x_skip = x
        for i in range(self.layer3_repeats):
            out_layer3 = self._backbone.layer3[i](self.relu(x))
            out_layer3_skip = self.layer3_skip[i](self.relu(x_skip))

            x = out_layer3 + self.down3[i](self.relu(out_layer3_skip))
            x_skip = out_layer3_skip + self.upscale(self.compression3[i](self.relu(out_layer3)), height_output, width_output)

        # save for auxiliary head
        if self.use_aux_heads:
            temp = x_skip

        out_layer4 = self._backbone.layer4(self.relu(x))
        out_layer4_skip = self.layer4_skip(self.relu(x_skip))

        x = out_layer4 + self.down4(self.relu(out_layer4_skip))
        x_skip = out_layer4_skip + self.upscale(self.compression4(self.relu(out_layer4)), height_output, width_output)

        out_layer5_skip = self.layer5_skip(self.relu(x_skip))

        if self.classification_mode:
            x_skip = self.high_to_low_fusion(self.relu(out_layer5_skip))
            x = self.layer5(self.relu(x))
            x = self.average_pool(x + x_skip)
            x = self.fc(x.squeeze())
            return x
        else:
            x = self.upscale(self.spp(self.layer5(self.relu(x))), height_output, width_output)

            x = self.final_layer(x + out_layer5_skip)

            if self.use_aux_heads:
                x_extra = self.seghead_extra(temp)
                return x, x_extra
            else:
                return x

    def replace_head(self, new_num_classes=None, new_head=None, new_aux_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_aux_head is not None:
            self.seghead_extra = new_aux_head
        if new_head is not None:
            self.final_layer = new_head
        else:
            self.final_layer = SegmentHead(
                self.highres_planes * self.layer5_bottleneck_expansion, self.head_width, new_num_classes, 8, inter_mode=self.segmentation_inter_mode
            )
            if self.use_aux_heads:
                self.seghead_extra = SegmentHead(self.highres_planes, self.head_width, new_num_classes, 8, inter_mode=self.segmentation_inter_mode)

    def _remove_auxiliary_heads(self):
        if hasattr(self, "seghead_extra"):
            del self.seghead_extra

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        Custom param groups for training:
            - Different lr for backbone and the rest, if `multiply_head_lr` key is in `training_params`.
        """
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        multiply_lr_params, no_multiply_params = self._separate_lr_multiply_params()
        param_groups = [
            {"named_params": no_multiply_params, "lr": lr, "name": "no_multiply_params"},
            {"named_params": multiply_lr_params, "lr": lr * multiply_head_lr, "name": "multiply_lr_params"},
        ]
        return param_groups

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        for param_group in param_groups:
            param_group["lr"] = lr
            if param_group["name"] == "multiply_lr_params":
                param_group["lr"] *= multiply_head_lr
        return param_groups

    def _separate_lr_multiply_params(self):
        """
        Separate backbone params from the rest.
        :return: iterators of groups named_parameters.
        """
        backbone_names = [n for n, p in self.backbone.named_parameters()]
        multiply_lr_params, no_multiply_params = {}, {}
        for name, param in self.named_parameters():
            if name in backbone_names:
                no_multiply_params[name] = param
            else:
                multiply_lr_params[name] = param
        return multiply_lr_params.items(), no_multiply_params.items()


class DDRNetCustom(DDRNet):
    def __init__(self, arch_params: HpmStruct):
        """Parse arch_params and translate the parameters to build the original DDRNet architecture"""
        if get_param(arch_params, "aux_heads") is not None:
            message = "arch_params.aux_heads is deprecated in 3.1.1 and will be removed in 3.2.0."
            if get_param(arch_params, "use_aux_heads") is not None:
                message += "\n using arch_params.use_aux_heads instead."

            else:
                message += "\n use arch_params.use_aux_heads instead."
            warnings.warn(message, DeprecationWarning)
            use_aux_heads = get_param(arch_params, "aux_heads")
        else:
            use_aux_heads = get_param(arch_params, "use_aux_heads")
        super().__init__(
            backbone=arch_params.backbone,
            additional_layers=arch_params.additional_layers,
            upscale_module=arch_params.upscale_module,
            num_classes=arch_params.num_classes,
            highres_planes=arch_params.highres_planes,
            spp_width=arch_params.spp_planes,
            head_width=arch_params.head_planes,
            use_aux_heads=use_aux_heads,
            ssp_inter_mode=arch_params.ssp_inter_mode,
            segmentation_inter_mode=arch_params.segmentation_inter_mode,
            skip_block=arch_params.skip_block,
            layer5_block=arch_params.layer5_block,
            layer5_bottleneck_expansion=arch_params.layer5_bottleneck_expansion,
            classification_mode=arch_params.classification_mode,
            spp_kernel_sizes=arch_params.spp_kernel_sizes,
            spp_strides=arch_params.spp_strides,
            layer3_repeats=arch_params.layer3_repeats,
        )


DEFAULT_DDRNET_23_PARAMS = {
    "input_channels": 3,
    "block": BasicResNetBlock,
    "skip_block": BasicResNetBlock,
    "layer5_block": Bottleneck,
    "layer5_bottleneck_expansion": 2,
    "layers": [2, 2, 2, 2, 1, 2, 2, 1],
    "upscale_module": UpscaleOnline(),
    "planes": 64,
    "highres_planes": 128,
    "head_planes": 128,
    "use_aux_heads": False,
    "segmentation_inter_mode": "bilinear",
    "classification_mode": False,
    "spp_planes": 128,
    "ssp_inter_mode": "bilinear",
    "spp_kernel_sizes": [1, 5, 9, 17, 0],
    "spp_strides": [1, 2, 4, 8, 0],
    "layer3_repeats": 1,
}

DEFAULT_DDRNET_23_SLIM_PARAMS = {
    **DEFAULT_DDRNET_23_PARAMS,
    "planes": 32,
    "highres_planes": 64,
    "head_planes": 64,
}

DEFAULT_DDRNET_39_PARAMS = {**DEFAULT_DDRNET_23_PARAMS, "layers": [3, 4, 3, 3, 1, 3, 3, 1], "head_planes": 256, "layer3_repeats": 2}


@register_model(Models.DDRNET_39)
class DDRNet39(DDRNetCustom):
    def __init__(self, arch_params: HpmStruct):
        _arch_params = HpmStruct(**DEFAULT_DDRNET_39_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        # BUILD THE BACKBONE AND INSERT TO THE _arch_params
        backbone_layers, _arch_params.additional_layers = _arch_params.layers[:4], _arch_params.layers[4:]
        _arch_params.backbone = BasicDDRBackBone(
            block=_arch_params.block,
            width=_arch_params.planes,
            layers=backbone_layers,
            input_channels=_arch_params.input_channels,
            layer3_repeats=_arch_params.layer3_repeats,
        )
        super().__init__(_arch_params)


@register_model(Models.DDRNET_23)
class DDRNet23(DDRNetCustom):
    def __init__(self, arch_params: HpmStruct):
        _arch_params = HpmStruct(**DEFAULT_DDRNET_23_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        # BUILD THE BACKBONE AND INSERT TO THE _arch_params
        backbone_layers, _arch_params.additional_layers = _arch_params.layers[:4], _arch_params.layers[4:]
        _arch_params.backbone = BasicDDRBackBone(
            block=_arch_params.block,
            width=_arch_params.planes,
            layers=backbone_layers,
            input_channels=_arch_params.input_channels,
            layer3_repeats=_arch_params.layer3_repeats,
        )
        super().__init__(_arch_params)


@register_model(Models.DDRNET_23_SLIM)
class DDRNet23Slim(DDRNetCustom):
    def __init__(self, arch_params: HpmStruct):
        _arch_params = HpmStruct(**DEFAULT_DDRNET_23_SLIM_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        # BUILD THE BACKBONE AND INSERT TO THE _arch_params
        backbone_layers, _arch_params.additional_layers = _arch_params.layers[:4], _arch_params.layers[4:]
        _arch_params.backbone = BasicDDRBackBone(
            block=_arch_params.block,
            width=_arch_params.planes,
            layers=backbone_layers,
            input_channels=_arch_params.input_channels,
            layer3_repeats=_arch_params.layer3_repeats,
        )
        super().__init__(_arch_params)


@register_model(Models.CUSTOM_DDRNET_23)
class AnyBackBoneDDRNet23(DDRNetCustom):
    def __init__(self, arch_params: HpmStruct):
        _arch_params = HpmStruct(**DEFAULT_DDRNET_23_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        assert len(_arch_params.layers) == 4 or len(_arch_params.layers) == 8, "The length of 'arch_params.layers' must be 4 or 8"
        # TAKE THE LAST 4 NUMBERS AS THE ADDITIONAL LAYERS SPECIFICATION
        _arch_params.additional_layers = _arch_params.layers[-4:]
        assert hasattr(_arch_params, "backbone"), "AnyBackBoneDDRNet_23 requires having a backbone in arch_params"
        if hasattr(_arch_params, "input_channels"):
            assert (
                _arch_params.backbone.input_channels == _arch_params.input_channels
            ), "'input_channels' was given in arch_params with a different value than existing in the backbone"
        super().__init__(_arch_params)
