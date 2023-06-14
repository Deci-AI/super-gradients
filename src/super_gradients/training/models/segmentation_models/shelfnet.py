"""
Shelfnet

paper: https://arxiv.org/abs/1811.11254
based on: https://github.com/juntang-zhuang/ShelfNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils import HpmStruct
from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.models.classification_models.resnet import BasicResNetBlock, ResNet, Bottleneck


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.fcn = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.fcn(x)


class ShelfBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, dropout: float = 0.25):
        """
        S-Block implementation from the ShelfNet paper
            :param in_planes:   input planes
            :param planes:      output planes
            :param stride:      convolution stride
            :param dropout:     dropout percentage
        """
        super().__init__()
        if in_planes != planes:
            self.conv0 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
            self.relu0 = nn.ReLU(inplace=True)

        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.in_planes != self.planes:
            x = self.conv0(x)
            x = self.relu0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = out + x

        return self.relu2(out)


class ShelfResNetBackBone(ResNet):
    """
    ShelfResNetBackBone - A class that Inherits from the original ResNet class and manipulates the forward pass,
                          to create a backbone for the ShelfNet architecture
    """

    def __init__(self, block, num_blocks, num_classes=10, width_mult=1):
        super().__init__(block=block, num_blocks=num_blocks, num_classes=num_classes, width_mult=width_mult, backbone_mode=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        feat4 = self.layer1(out)  # 1/4
        feat8 = self.layer2(feat4)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat4, feat8, feat16, feat32


class ShelfResNetBackBone18(ShelfResNetBackBone):
    def __init__(self, num_classes: int):
        super().__init__(BasicResNetBlock, [2, 2, 2, 2], num_classes=num_classes)


class ShelfResNetBackBone34(ShelfResNetBackBone):
    def __init__(self, num_classes: int):
        super().__init__(BasicResNetBlock, [3, 4, 6, 3], num_classes=num_classes)


class ShelfResNetBackBone503343(ShelfResNetBackBone):
    def __init__(self, num_classes: int):
        super().__init__(Bottleneck, [3, 3, 4, 3], num_classes=num_classes)


class ShelfResNetBackBone50(ShelfResNetBackBone):
    def __init__(self, num_classes: int):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


class ShelfResNetBackBone101(ShelfResNetBackBone):
    def __init__(self, num_classes: int):
        super().__init__(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


class ShelfNetModuleBase(SgModule):
    """
    ShelfNetModuleBase - Base class for the different Modules of the ShelfNet Architecture
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ConvBNReLU(ShelfNetModuleBase):
    def __init__(self, in_chan: int, out_chan: int, ks: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class DecoderBase(ShelfNetModuleBase):
    def __init__(self, planes: int, layers: int, kernel: int = 3, block=ShelfBlock):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel
        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes)

        # CREATE MODULE FOR BOTTOM BLOCK
        self.bottom = block(planes * (2 ** (layers - 1)), planes * (2 ** (layers - 1)))

        # CREATE MODULE LIST FOR UP BRANCH
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()

    def forward(self, x):
        raise NotImplementedError


class DecoderHW(DecoderBase):
    """
    DecoderHW - The Decoder for the Heavy-Weight ShelfNet Architecture
    """

    def __init__(self, planes, layers, block=ShelfBlock, *args, **kwargs):
        super().__init__(planes=planes, layers=layers, block=block, *args, **kwargs)

        for i in range(0, layers - 1):
            self.up_conv_list.append(
                nn.ConvTranspose2d(
                    planes * 2 ** (layers - 1 - i), planes * 2 ** max(0, layers - i - 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
                )
            )
            self.up_dense_list.append(block(planes * 2 ** max(0, layers - i - 2), planes * 2 ** max(0, layers - i - 2)))

    def forward(self, x):
        # BOTTOM BRANCH
        out = self.bottom(x[-1])
        bottom = out

        # UP BRANCH
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out) + x[self.layers - j - 2]
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class DecoderLW(DecoderBase):
    """
    DecoderLW - The Decoder for the Light-Weight ShelfNet Architecture
    """

    def __init__(self, planes, layers, block=ShelfBlock, *args, **kwargs):
        super().__init__(planes=planes, layers=layers, block=block, *args, **kwargs)

        for i in range(0, layers - 1):
            self.up_conv_list.append(AttentionRefinementModule(planes * 2 ** (layers - 1 - i), planes * 2 ** max(0, layers - i - 2)))
            self.up_dense_list.append(ConvBNReLU(in_chan=planes * 2 ** max(0, layers - i - 2), out_chan=planes * 2 ** max(0, layers - i - 2), ks=3, stride=1))

    def forward(self, x):
        # BOTTOM BRANCH
        out = self.bottom(x[-1])
        bottom = out

        # UP BRANCH
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out)
            out_interpolate = F.interpolate(out, (out.size(2) * 2, out.size(3) * 2), mode="nearest")
            out = out_interpolate + x[self.layers - j - 2]
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class LadderBlockBase(ShelfNetModuleBase):
    def __init__(self, planes: int, layers: int, kernel: int = 3, block=ShelfBlock):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes)

        # CREATE MODULE LIST FOR DOWN BRANCH
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_module_list.append(block(planes * (2**i), planes * (2**i)))

        # USE STRIDED CONV INSTEAD OF POOLING
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_conv_list.append(nn.Conv2d(planes * 2**i, planes * 2 ** (i + 1), stride=2, kernel_size=kernel, padding=self.padding))

        # CREATE MODULE FOR BOTTOM BLOCK
        self.bottom = block(planes * (2 ** (layers - 1)), planes * (2 ** (layers - 1)))

        # CREATE MODULE LIST FOR UP BRANCH
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()

    def forward(self, x):
        raise NotImplementedError


class LadderBlockHW(LadderBlockBase):
    """
    LadderBlockHW - LadderBlock for the Heavy-Weight ShelfNet Architecture
    """

    def __init__(self, planes, layers, block=ShelfBlock, *args, **kwargs):
        super().__init__(planes=planes, layers=layers, block=block, *args, **kwargs)

        for i in range(0, layers - 1):
            self.up_conv_list.append(
                nn.ConvTranspose2d(
                    planes * 2 ** (layers - i - 1), planes * 2 ** max(0, layers - i - 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
                )
            )

            self.up_dense_list.append(block(planes * 2 ** max(0, layers - i - 2), planes * 2 ** max(0, layers - i - 2)))

    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # down branch
        for i in range(0, self.layers - 1):
            out = out + x[-i - 1]
            out = self.down_module_list[i](out)
            down_out.append(out)

            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out) + down_out[self.layers - j - 2]
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class LadderBlockLW(LadderBlockBase):
    """
    LadderBlockLW - LadderBlock for the Light-Weight ShelfNet Architecture
    """

    def __init__(self, planes, layers, block=ShelfBlock, *args, **kwargs):
        super().__init__(planes=planes, layers=layers, block=block, *args, **kwargs)

        for i in range(0, layers - 1):
            self.up_conv_list.append(AttentionRefinementModule(planes * 2 ** (layers - 1 - i), planes * 2 ** max(0, layers - i - 2)))
            self.up_dense_list.append(ConvBNReLU(in_chan=planes * 2 ** max(0, layers - i - 2), out_chan=planes * 2 ** max(0, layers - i - 2), ks=3, stride=1))

    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # DOWN BRANCH
        for i in range(0, self.layers - 1):
            out = out + x[-i - 1]
            out = self.down_module_list[i](out)
            down_out.append(out)

            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # BOTTOM BRANCH
        out = self.bottom(out)
        bottom = out

        # UP BRANCH
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out)
            out = F.interpolate(out, (out.size(2) * 2, out.size(3) * 2), mode="nearest") + down_out[self.layers - j - 2]
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class NetOutput(ShelfNetModuleBase):
    def __init__(self, in_chan: int, mid_chan: int, num_classes: int):
        super(NetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, num_classes, kernel_size=3, bias=False, padding=1)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ShelfNetBase(ShelfNetModuleBase):
    """
    ShelfNetBase - ShelfNet Base Generic Architecture
    """

    def __init__(
        self,
        backbone: ShelfResNetBackBone,
        planes: int,
        layers: int,
        num_classes: int = 21,
        image_size: int = 512,
        net_output_mid_channels_num: int = 64,
        arch_params: HpmStruct = None,
    ):
        self.num_classes = arch_params.num_classes if (arch_params and hasattr(arch_params, "num_classes")) else num_classes
        self.image_size = arch_params.image_size if (arch_params and hasattr(arch_params, "image_size")) else image_size

        super().__init__()
        self.net_output_mid_channels_num = net_output_mid_channels_num
        self.backbone = backbone(self.num_classes)
        self.layers = layers
        self.planes = planes

        # INITIALIZE WITH AUXILARY HEAD OUTPUTS ONN -> TURN IT OFF TO RUN A FORWARD PASS WITHOUT THE AUXILARY HEADS
        self.auxilary_head_outputs = True

        # DECODER AND LADDER SHOULD BE IMPLEMENTED BY THE INHERITING CLASS
        self.decoder = None
        self.ladder = None

        # BUILD THE CONV_OUT LIST BASED ON THE AMOUNT OF LAYERS IN THE SHELFNET
        self.conv_out_list = torch.nn.ModuleList()

    def forward(self, x):
        raise NotImplementedError

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        """
        update_optimizer_for_param_groups - Updates the specific parameters with different LR
        """
        # LEARNING RATE FOR THE BACKBONE IS lr
        param_groups[0]["lr"] = lr
        for i in range(1, len(param_groups)):
            # LEARNING RATE FOR OTHER SHELFNET PARAMS IS lr * 10
            param_groups[i]["lr"] = lr * 10

        return param_groups


class ShelfNetHW(ShelfNetBase):
    """
    ShelfNetHW - Heavy-Weight Version of ShelfNet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ladder = LadderBlockHW(planes=self.net_output_mid_channels_num, layers=self.layers)
        self.decoder = DecoderHW(planes=self.net_output_mid_channels_num, layers=self.layers)
        self.se_layer = nn.Linear(self.net_output_mid_channels_num * 2**3, self.num_classes)
        self.aux_head = FCNHead(1024, self.num_classes)
        self.final = nn.Conv2d(self.net_output_mid_channels_num, self.num_classes, 1)

        # THE MID CHANNELS NUMBER OF THE NET OUTPUT BLOCK
        net_out_planes = self.planes
        mid_channels_num = self.net_output_mid_channels_num

        # INITIALIZE THE conv_out_list
        for i in range(self.layers):
            self.conv_out_list.append(ConvBNReLU(in_chan=net_out_planes, out_chan=mid_channels_num, ks=1, padding=0))

            mid_channels_num *= 2
            net_out_planes *= 2

    def forward(self, x):
        image_size = x.size()[2:]

        backbone_features_list = list(self.backbone(x))
        conv_bn_relu_results_list = []

        for feature, conv_bn_relu in zip(backbone_features_list, self.conv_out_list):
            out = conv_bn_relu(feature)
            conv_bn_relu_results_list.append(out)

        decoder_out_list = self.decoder(conv_bn_relu_results_list)
        ladder_out_list = self.ladder(decoder_out_list)

        preds = [self.final(ladder_out_list[-1])]

        # SE_LOSS ENCODING
        enc = F.max_pool2d(ladder_out_list[0], kernel_size=ladder_out_list[0].size()[2:])
        enc = torch.squeeze(enc, -1)
        enc = torch.squeeze(enc, -1)
        se = self.se_layer(enc)
        preds.append(se)

        # UP SAMPLING THE TOP LAYER FOR PREDICTION
        preds[0] = F.interpolate(preds[0], image_size, mode="bilinear", align_corners=True)

        # AUXILARY HEAD OUTPUT (ONLY RELEVANT FOR LOSS CALCULATION) - USE self.auxilary_head_outputs=FALSE FOR INFERENCE
        if self.auxilary_head_outputs or self.training:
            aux_out = self.aux_head(backbone_features_list[2])
            aux_out = F.interpolate(aux_out, image_size, mode="bilinear", align_corners=True)
            preds.append(aux_out)

            return tuple(preds)
        else:
            return preds[0]

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        initialize_optimizer_for_model_param_groups - Initializes the weights of the optimizer
                                                      Initializes the Backbone, the Output and the Auxilary Head
                                                      differently
            :param optimizer_cls:   The nn.optim (optimizer class) to initialize
            :param lr:              lr to set for the optimizer
            :param training_params:
            :return: list of dictionaries with named params and optimizer attributes
        """
        # OPTIMIZER PARAMETER GROUPS
        params_list = []

        # OPTIMIZE BACKBONE USING DIFFERENT LR
        params_list.append({"named_params": self.backbone.named_parameters(), "lr": lr})

        # OPTIMIZE MAIN SHELFNET ARCHITECTURE LAYERS
        params_list.append(
            {
                "named_params": list(self.ladder.named_parameters())
                + list(self.decoder.named_parameters())
                + list(self.se_layer.named_parameters())
                + list(self.conv_out_list.named_parameters())
                + list(self.final.named_parameters())
                + list(self.aux_head.named_parameters()),
                "lr": lr * 10,
            }
        )

        return params_list


class ShelfNetLW(ShelfNetBase):
    """
    ShelfNetLW - Light-Weight Implementation for ShelfNet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.net_output_list = nn.ModuleList()
        self.ladder = LadderBlockLW(planes=self.planes, layers=self.layers)
        self.decoder = DecoderLW(planes=self.planes, layers=self.layers)

    def forward(self, x):
        H, W = x.size()[2:]

        # SHELFNET LW ARCHITECTURE USES ONLY LAST 3 PARTIAL OUTPUTs OF THE BACKBONE'S 4 OUTPUT LAYERS
        backbone_features_tuple = self.backbone(x)[1:]

        if isinstance(self, ShelfNet18_LW):
            # FOR SHELFNET18 USE 1x1 CONVS AFTER THE BACKBONE'S FORWARD PASS TO MANIPULATE THE CHANNELS FOR THE DECODER
            conv_bn_relu_results_list = []

            for feature, conv_bn_relu in zip(backbone_features_tuple, self.conv_out_list):
                out = conv_bn_relu(feature)
                conv_bn_relu_results_list.append(out)

        else:
            # FOR SHELFNET34 THE CHANNELS ARE ALREADY ALIGNED
            conv_bn_relu_results_list = list(backbone_features_tuple)

        decoder_out_list = self.decoder(conv_bn_relu_results_list)
        ladder_out_list = self.ladder(decoder_out_list)

        # GET THE LAST ELEMENTS OF THE LADDER_BLOCK BASED ON THE AMOUNT OF SHELVES IN THE ARCHITECTURE AND REVERSE LIST
        feat_cp_list = list(reversed(ladder_out_list[(-1 * self.layers) :]))

        feat_out = self.net_output_list[0](feat_cp_list[0])
        feat_out = F.interpolate(feat_out, (H, W), mode="bilinear", align_corners=True)

        if self.auxilary_head_outputs or self.training:
            features_out_list = [feat_out]
            for conv_output_layer, feat_cp in zip(self.net_output_list[1:], feat_cp_list[1:]):
                feat_out_res = conv_output_layer(feat_cp)
                feat_out_res = F.interpolate(feat_out_res, (H, W), mode="bilinear", align_corners=True)
                features_out_list.append(feat_out_res)

            return tuple(features_out_list)

        else:
            # THIS DOES NOT CALCULATE THE AUXILARY HEADS THAT ARE CRITICAL FOR THE LOSS (USED MAINLY FOR INFERENCE)
            return feat_out

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        initialize_optimizer_for_model_param_groups - Initializes the optimizer group params, with 10x learning rate
                                                      for all but the backbone

            :param lr:              lr to set for the backbone
            :param training_params:
            :return: list of dictionaries with named params and optimizer attributes
        """
        # OPTIMIZER PARAMETER GROUPS
        params_list = []

        # OPTIMIZE BACKBONE USING DIFFERENT LR
        params_list.append({"named_params": self.backbone.named_parameters(), "lr": lr})

        # OPTIMIZE MAIN SHELFNET ARCHITECTURE LAYERS
        params_list.append(
            {
                "named_params": list(self.ladder.named_parameters()) + list(self.decoder.named_parameters()) + list(self.conv_out_list.named_parameters()),
                "lr": lr * 10,
            }
        )

        return params_list


@register_model(Models.SHELFNET18_LW)
class ShelfNet18_LW(ShelfNetLW):
    def __init__(self, *args, **kwargs):
        super().__init__(backbone=ShelfResNetBackBone18, planes=64, layers=3, *args, **kwargs)

        # INITIALIZE THE net_output_list AND THE conv_out LIST
        out_planes = self.planes
        for i in range(self.layers):
            # THE MID CHANNELS NUMBER OF THE NET OUTPUT BLOCK
            mid_channels_num = self.planes if i == 0 else self.net_output_mid_channels_num

            self.net_output_list.append(NetOutput(out_planes, mid_channels_num, self.num_classes))

            self.conv_out_list.append(ConvBNReLU(out_planes * 2, out_planes, ks=1, stride=1, padding=0))

            out_planes *= 2


@register_model(Models.SHELFNET34_LW)
class ShelfNet34_LW(ShelfNetLW):
    def __init__(self, *args, **kwargs):
        super().__init__(backbone=ShelfResNetBackBone34, planes=128, layers=3, *args, **kwargs)

        # INITIALIZE THE net_output_list
        net_out_planes = self.planes
        for i in range(self.layers):
            # IF IT'S THE FIRST LAYER THAN THE MID-CHANNELS NUM IS ACTUALLY self.planes
            mid_channels_num = self.planes if i == 0 else self.net_output_mid_channels_num
            self.net_output_list.append(NetOutput(net_out_planes, mid_channels_num, self.num_classes))

            net_out_planes *= 2


@register_model(Models.SHELFNET50_3343)
class ShelfNet503343(ShelfNetHW):
    def __init__(self, *args, **kwargs):
        super().__init__(backbone=ShelfResNetBackBone503343, planes=256, layers=4, *args, **kwargs)


@register_model(Models.SHELFNET50)
class ShelfNet50(ShelfNetHW):
    def __init__(self, *args, **kwargs):
        super().__init__(backbone=ShelfResNetBackBone50, planes=256, layers=4, *args, **kwargs)


@register_model(Models.SHELFNET101)
class ShelfNet101(ShelfNetHW):
    def __init__(self, *args, **kwargs):
        super().__init__(backbone=ShelfResNetBackBone101, planes=256, layers=4, *args, **kwargs)
