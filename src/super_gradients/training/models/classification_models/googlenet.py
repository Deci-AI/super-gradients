"""
Googlenet code based on https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html

"""
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.models.sg_module import SgModule

GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["log_", "aux_logits2", "aux_logits1"])


class GoogLeNet(SgModule):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True, backbone_mode=False, dropout=0.3):
        super(GoogLeNet, self).__init__()

        self.num_classes = num_classes
        self.backbone_mode = backbone_mode

        self.aux_logits = aux_logits
        self.dropout_p = dropout

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if not self.backbone_mode:
            self.dropout = nn.Dropout(self.dropout_p)
            self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats

                x = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(x.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1 = None
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2 = None
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        if not self.backbone_mode:
            x = self.dropout(x)
            x = self.fc(x)
        # N x num_classes
        return x, aux2, aux1

    def forward(self, x):
        x, aux1, aux2 = self._forward(x)
        if self.training and self.aux_logits:
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def load_state_dict(self, state_dict, strict=True):
        """
        load_state_dict - Overloads the base method and calls it to load a modified dict for usage as a backbone
        :param state_dict:  The state_dict to load
        :param strict:      strict loading (see super() docs)
        """
        pretrained_model_weights_dict = state_dict.copy()

        if self.backbone_mode:
            # FIRST LET'S POP THE LAST TWO LAYERS - NO NEED TO LOAD THEIR VALUES SINCE THEY ARE IRRELEVANT AS A BACKBONE
            pretrained_model_weights_dict.popitem()
            pretrained_model_weights_dict.popitem()

            pretrained_backbone_weights_dict = OrderedDict()
            for layer_name, weights in pretrained_model_weights_dict.items():
                # GET THE LAYER NAME WITHOUT THE 'module.' PREFIX
                name_without_module_prefix = layer_name.split("module.")[1]

                # MAKE SURE THESE ARE NOT THE FINAL LAYERS
                pretrained_backbone_weights_dict[name_without_module_prefix] = weights
            c_temp = torch.nn.Linear(1024, self.num_classes)
            torch.nn.init.xavier_uniform(c_temp.weight)
            pretrained_backbone_weights_dict["fc.weight"] = c_temp.weight
            pretrained_backbone_weights_dict["fc.bias"] = c_temp.bias
            # RETURNING THE UNMODIFIED/MODIFIED STATE DICT DEPENDING ON THE backbone_mode VALUE
            super().load_state_dict(pretrained_backbone_weights_dict, strict)
        else:
            super().load_state_dict(pretrained_model_weights_dict, strict)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1))

        self.branch3 = nn.Sequential(conv_block(in_channels, ch5x5red, kernel_size=1), conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1))

        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True), conv_block(in_channels, pool_proj, kernel_size=1))

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@register_model(Models.GOOGLENET_V1)
class GoogleNetV1(GoogLeNet):
    def __init__(self, arch_params):
        super(GoogleNetV1, self).__init__(aux_logits=False, num_classes=arch_params.num_classes, dropout=arch_params.dropout)
