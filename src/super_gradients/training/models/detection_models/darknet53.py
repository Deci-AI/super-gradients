from torch import nn

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils import get_param


def create_conv_module(in_channels, out_channels, kernel_size=3, stride=1):
    padding = (kernel_size - 1) // 2
    nn_sequential_module = nn.Sequential()
    nn_sequential_module.add_module("Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
    nn_sequential_module.add_module("BatchNorm2d", nn.BatchNorm2d(out_channels))
    nn_sequential_module.add_module("LeakyRelu", nn.LeakyReLU())

    return nn_sequential_module


# Residual block
class DarkResidualBlock(nn.Module):
    """
    DarkResidualBlock - The Darknet Residual Block
    """

    def __init__(self, in_channels, shortcut=True):
        super(DarkResidualBlock, self).__init__()
        self.shortcut = shortcut
        reduced_channels = int(in_channels / 2)

        self.layer1 = create_conv_module(in_channels, reduced_channels, kernel_size=1)
        self.layer2 = create_conv_module(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual if self.shortcut else out
        return out


class Darknet53Base(SgModule):
    def __init__(self):
        super(Darknet53Base, self).__init__()
        # THE MODULES LIST IS APPROACHABLE FROM "OUTSIDE THE CLASS - SO WE CAN CHANGE IT'S STRUCTURE"
        self.modules_list = nn.ModuleList()
        self.modules_list.append(create_conv_module(3, 32))  # 0
        self.modules_list.append(create_conv_module(32, 64, stride=2))  # 1
        self.modules_list.append(self._make_layer(DarkResidualBlock, in_channels=64, num_blocks=1))  # 2
        self.modules_list.append(create_conv_module(64, 128, stride=2))  # 3
        self.modules_list.append(self._make_layer(DarkResidualBlock, in_channels=128, num_blocks=2))  # 4
        self.modules_list.append(create_conv_module(128, 256, stride=2))  # 5
        self.modules_list.append(self._make_layer(DarkResidualBlock, in_channels=256, num_blocks=8))  # 6
        self.modules_list.append(create_conv_module(256, 512, stride=2))  # 7
        self.modules_list.append(self._make_layer(DarkResidualBlock, in_channels=512, num_blocks=8))  # 8
        self.modules_list.append(create_conv_module(512, 1024, stride=2))  # 9
        self.modules_list.append(self._make_layer(DarkResidualBlock, in_channels=1024, num_blocks=4))  # 10

    def forward(self, x):
        out = x
        for i, module in enumerate(self.modules_list):
            out = self.modules_list[i](out)

        return out

    def _make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


@register_model(Models.DARKNET53)
class Darknet53(Darknet53Base):
    def __init__(self, arch_params=None, backbone_mode=True, num_classes=None):
        super(Darknet53, self).__init__()

        # IN ORDER TO ALLOW PASSING PARAMETERS WITH ARCH_PARAMS BUT NOT BREAK YOLOV3 INTEGRATION
        self.backbone_mode = get_param(arch_params, "backbone_mode", backbone_mode)
        self.num_classes = get_param(arch_params, "num_classes", num_classes)

        if not self.backbone_mode:
            # IF NOT USED AS A BACKEND BUT AS A CLASSIFIER WE ADD THE CLASSIFICATION LAYERS
            if self.num_classes is not None:
                nn_sequential_block = nn.Sequential()
                nn_sequential_block.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
                nn_sequential_block.add_module("view", ViewModule(1024))
                nn_sequential_block.add_module("fc", nn.Linear(1024, self.num_classes))
                self.modules_list.append(nn_sequential_block)
            else:
                raise ValueError("num_classes must be specified to use Darknet53 as a classifier")

    def get_modules_list(self):
        return self.modules_list

    def forward(self, x):
        """
        forward - Forward pass on the modules list
            :param x: The input data
            :return: forward pass for backbone pass or classification pass
        """
        return super().forward(x)


# Residual block
class ViewModule(nn.Module):
    """
    Returns a reshaped version of the input, to be used in None-Backbone Mode
    """

    def __init__(self, features=1024):
        super(ViewModule, self).__init__()
        self.features = features

    def forward(self, x):
        return x.view(-1, self.features)
