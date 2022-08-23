import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from super_gradients.training.models.sg_module import SgModule

"""Densenet-BC model class, based on
"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
Code source: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
Performance reproducibility (4 GPUs):
 training params: {"max_epochs": 120, "lr_updates": [30, 60, 90, 100, 110], "lr_decay_factor": 0.1, "initial_lr": 0.025}
 dataset_params: {"batch_size": 64}
"""


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, input):  # noqa: F811
        prev_features = [input] if isinstance(input, Tensor) else input
        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(SgModule):
    def __init__(self, growth_rate: int, structure: list, num_init_features: int, bn_size: int, drop_rate: float,
                 num_classes: int):
        """
        :param growth_rate:         number of filter to add each layer (noted as 'k' in the paper)
        :param structure:           how many layers in each pooling block - sequentially
        :param num_init_features:   the number of filters to learn in the first convolutional layer
        :param bn_size:             multiplicative factor for the number of bottle neck layers
                                        (i.e. bn_size * k featurs in the bottleneck)
        :param drop_rate:           dropout rate after each dense layer
        :param num_classes:         number of classes in the classification task
        """
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(structure):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(structure) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def CustomizedDensnet(arch_params):
    return DenseNet(growth_rate=arch_params.growth_rate if hasattr(arch_params, "growth_rate") else 32,
                    structure=arch_params.structure if hasattr(arch_params, "structure") else [6, 12, 24, 16],
                    num_init_features=arch_params.num_init_features if hasattr(arch_params,
                                                                               "num_init_features") else 64,
                    bn_size=arch_params.bn_size if hasattr(arch_params, "bn_size") else 4,
                    drop_rate=arch_params.drop_rate if hasattr(arch_params, "drop_rate") else 0,
                    num_classes=arch_params.num_classes)


def densenet121(arch_params):
    return DenseNet(32, [6, 12, 24, 16], 64, 4, 0, arch_params.num_classes)


def densenet161(arch_params):
    return DenseNet(48, [6, 12, 36, 24], 96, 4, 0, arch_params.num_classes)


def densenet169(arch_params):
    return DenseNet(32, [6, 12, 32, 32], 64, 4, 0, arch_params.num_classes)


def densenet201(arch_params):
    return DenseNet(32, [6, 12, 48, 32], 64, 4, 0, arch_params.num_classes)
