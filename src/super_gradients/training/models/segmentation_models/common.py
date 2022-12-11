import torch.nn as nn
from super_gradients.modules import ConvBNReLU


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int, dropout: float):
        super(SegmentationHead, self).__init__()
        self.seg_head = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.seg_head(x)

    def replace_num_classes(self, num_classes: int):
        """
        This method replace the last Conv Classification layer to output a different number of classes.
        Note that the weights of the new layers are random initiated.
        """
        old_cls_conv = self.seg_head[-1]
        self.seg_head[-1] = nn.Conv2d(old_cls_conv.in_channels, num_classes, kernel_size=1, bias=False)
