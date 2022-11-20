"""MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
"""
import torch.nn as nn
import torch.nn.functional as F
from super_gradients.training.models.sg_module import SgModule


class Block(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(SgModule):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, 128, (128, 2), 256, (256, 2), 512, 512, 512, 512, 512, (512, 2), 1024, (1024, 2)]

    def __init__(self, num_classes=10, backbone_mode=False, up_to_layer=None, in_channels: int = 3):
        super(MobileNet, self).__init__()
        self.backbone_mode = backbone_mode
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, up_to_layer=up_to_layer if up_to_layer is not None else len(self.cfg))

        if not self.backbone_mode:
            self.linear = nn.Linear(self.cfg[-1], num_classes)

    def _make_layers(self, in_planes, up_to_layer):
        layers = []
        for x in self.cfg[:up_to_layer]:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param up_to_layer: forward through the net layers up to a specific layer. if None, run all layers
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)

        if not self.backbone_mode:
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

        return out
