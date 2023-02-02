import torch
import torch.nn as nn
import torch.nn.functional as F

up_kwargs = {"mode": "bilinear", "align_corners": True}


# from encoding.nn import SyncBatchNorm # FIXME - ORIGINAL CODE TORCH-ENCODING


class LadderBottleneck(nn.Module):
    """ResNet Bottleneck"""

    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert len(x) == len(y)
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LadderResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."
            Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    # def __init__(self, block, layers, num_classes=1000, dilated=False, norm_layer=SyncBatchNorm): # FIXME - ORIGINAL CODE
    def __init__(self, block, layers, num_classes=1000, dilated=False, norm_layer=nn.BatchNorm2d):  # FIXME - TIME MEASUREMENT CODE
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            import math

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class LadderNetBackBone503433(LadderResNet):
    def __init__(self, num_classes: int):
        super().__init__(LadderBottleneck, [3, 4, 3, 3], num_classes=num_classes)


class LadderNetBackBone50(LadderResNet):
    def __init__(self, num_classes: int):
        super().__init__(LadderBottleneck, [3, 4, 6, 3], num_classes=num_classes)


class LadderNetBackBone101(LadderResNet):
    def __init__(self, num_classes: int):
        super().__init__(LadderBottleneck, [3, 4, 23, 3], num_classes=num_classes)


class BaseNet(nn.Module):
    def __init__(
        self,
        nclass,
        backbone,
        aux,
        se_loss,
        dilated=True,
        norm_layer=None,
        base_size=576,
        crop_size=608,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        root="~/.encoding/models",
    ):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        self.image_size = self.crop_size

        # copying modules from pretrained models
        if backbone == "resnet50":
            self.backbone = LadderNetBackBone50(num_classes=1000)
        elif backbone == "resnet50_3433":
            self.backbone = LadderNetBackBone503433(num_classes=1000)
        elif backbone == "resnet101":
            self.backbone = LadderNetBackBone101(num_classes=1000)
        # elif backbone == 'resnet152':
        #     self.pretrained = resnet.resnet152(pretrained=True, dilated=dilated,
        #                                        norm_layer=norm_layer, root=root)
        # elif backbone == 'resnet18':
        #     self.pretrained = resnet.resnet18(pretrained=True, dilated=dilated,
        #                                        norm_layer=norm_layer, root=root)
        # elif backbone == 'resnet34':
        #     self.pretrained = resnet.resnet34(pretrained=True, dilated=dilated,
        #                                        norm_layer=norm_layer, root=root)
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)

        return c1, c2, c3, c4

    # def evaluate(self, x, target=None):
    #     pred = self.forward(x)
    #     if isinstance(pred, (tuple, list)):
    #         pred = pred[0]
    #     if target is None:
    #         return pred
    #     correct, labeled = batch_pix_accuracy(pred.data, target.data)
    #     inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
    #     return correct, labeled, inter, union


drop = 0.25


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(BasicBlock, self).__init__()
        if inplanes != planes:
            self.conv0 = conv3x3(inplanes, planes, rate)

        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=drop)

    def forward(self, x):
        if self.inplanes != self.planes:
            x = self.conv0(x)
            x = F.relu(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop(out)

        out1 = self.conv1(out)
        out1 = self.bn2(out1)
        # out1 = self.relu(out1)

        out2 = out1 + x

        return F.relu(out2)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Initial_LadderBlock(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.in_bn = nn.BatchNorm2d(planes)
        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_module_list.append(block(planes * (2**i), planes * (2**i)))

        # use strided conv instead of poooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_conv_list.append(nn.Conv2d(planes * 2**i, planes * 2 ** (i + 1), stride=2, kernel_size=kernel, padding=self.padding))

        # create module for bottom block
        self.bottom = block(planes * (2**layers), planes * (2**layers))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(
                nn.ConvTranspose2d(
                    in_channels=planes * 2 ** (layers - i),
                    out_channels=planes * 2 ** max(0, layers - i - 1),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                )
            )
            self.up_dense_list.append(block(planes * 2 ** max(0, layers - i - 1), planes * 2 ** max(0, layers - i - 1)))

    def forward(self, x):
        out = self.inconv(x)
        out = self.in_bn(out)
        out = F.relu(out)

        down_out = []
        # down branch
        for i in range(0, self.layers):
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

        for j in range(0, self.layers):
            out = self.up_conv_list[j](out) + down_out[self.layers - j - 1]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class Decoder(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel
        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes)
        # create module for bottom block
        self.bottom = block(planes * (2 ** (layers - 1)), planes * (2 ** (layers - 1)))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.up_conv_list.append(
                nn.ConvTranspose2d(
                    planes * 2 ** (layers - 1 - i), planes * 2 ** max(0, layers - i - 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
                )
            )
            self.up_dense_list.append(block(planes * 2 ** max(0, layers - i - 2), planes * 2 ** max(0, layers - i - 2)))

    def forward(self, x):
        # bottom branch
        out = self.bottom(x[-1])
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out) + x[self.layers - j - 2]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class LadderBlock(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_module_list.append(block(planes * (2**i), planes * (2**i)))

        # use strided conv instead of pooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_conv_list.append(nn.Conv2d(planes * 2**i, planes * 2 ** (i + 1), stride=2, kernel_size=kernel, padding=self.padding))

        # create module for bottom block
        self.bottom = block(planes * (2 ** (layers - 1)), planes * (2 ** (layers - 1)))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
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
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class Final_LadderBlock(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.block = LadderBlock(planes, layers, kernel=kernel, block=block)

    def forward(self, x):
        out = self.block(x)
        return out[-1]


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.conv5(x)


class LadderNet(BaseNet):
    def __init__(
        self,
        nclass,
        backbone,
        aux=True,
        se_loss=True,
        lateral=False,
        arch_params=None,
        # norm_layer=SyncBatchNorm, dilated=False, **kwargs):  # FIXME - ORIGINAL CODE TORCH-ENCODING
        norm_layer=nn.BatchNorm2d,
        dilated=False,
        **kwargs,
    ):  # FIXME - TIME MEASUREMENT CODE
        super().__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, dilated=dilated, **kwargs)
        self.head = LadderHead(
            base_inchannels=256, base_outchannels=64, out_channels=nclass, norm_layer=norm_layer, se_loss=se_loss, nclass=nclass, up_kwargs=self._up_kwargs
        )
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(features))

        x[0] = F.upsample(x[0], imsize, **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class LadderHead(nn.Module):
    def __init__(self, base_inchannels, base_outchannels, out_channels, norm_layer, se_loss, nclass, up_kwargs):
        super(LadderHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=base_inchannels, out_channels=base_outchannels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=base_inchannels * 2, out_channels=base_outchannels * 2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=base_inchannels * 2**2, out_channels=base_outchannels * 2**2, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=base_inchannels * 2**3, out_channels=base_outchannels * 2**3, kernel_size=1, bias=False)

        self.bn1 = norm_layer(base_outchannels)
        self.bn2 = norm_layer(base_outchannels * 2)
        self.bn3 = norm_layer(base_outchannels * 2**2)
        self.bn4 = norm_layer(base_outchannels * 2**3)

        self.decoder = Decoder(planes=base_outchannels, layers=4)
        self.ladder = LadderBlock(planes=base_outchannels, layers=4)
        self.final = nn.Conv2d(base_outchannels, out_channels, 1)

        self.se_loss = se_loss

        if self.se_loss:
            self.selayer = nn.Linear(base_outchannels * 2**3, nclass)

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = F.relu(out1)

        out2 = self.conv2(x2)
        out2 = self.bn2(out2)
        out2 = F.relu(out2)

        out3 = self.conv3(x3)
        out3 = self.bn3(out3)
        out3 = F.relu(out3)

        out4 = self.conv4(x4)
        out4 = self.bn4(out4)
        out4 = F.relu(out4)

        out = self.decoder([out1, out2, out3, out4])
        out = self.ladder(out)

        pred = [self.final(out[-1])]

        if self.se_loss:
            enc = F.max_pool2d(out[0], kernel_size=out[0].size()[2:])
            enc = torch.squeeze(enc, -1)
            enc = torch.squeeze(enc, -1)
            se = self.selayer(enc)
            pred.append(se)

        return pred


class LadderNet50(LadderNet):
    def __init__(self, *args, **kwargs):
        super().__init__(backbone="resnet50", nclass=21, *args, **kwargs)


class LadderNet503433(LadderNet):
    def __init__(self, *args, **kwargs):
        super().__init__(backbone="resnet50_3433", nclass=21, *args, **kwargs)


class LadderNet101(LadderNet):
    def __init__(self, *args, **kwargs):
        super().__init__(backbone="resnet101", nclass=21, *args, **kwargs)
