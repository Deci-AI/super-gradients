from typing import Union, List
from abc import abstractmethod, ABC

import torch
from torch import nn
from omegaconf.listconfig import ListConfig

from super_gradients.training.models import MobileNet, MobileNetV2, InvertedResidual
from super_gradients.training.utils.module_utils import MultiOutputModule


class BaseDetectionModule(nn.Module, ABC):
    """
    An interface for a module that is easy to integrate into a model with complex connections
    """

    def __init__(self, in_channels: Union[List[int], int], **kwargs):
        """
        :param in_channels: defines channels of tensor(s) that will be accepted by a module in forward
        """
        super().__init__()
        self.in_channels = in_channels

    @property
    @abstractmethod
    def out_channels(self) -> Union[List[int], int]:
        raise NotImplementedError()


class MobileNetV1Backbone(BaseDetectionModule):
    """MobileNetV1 backbone with an option to return output of any layer"""

    def __init__(self, in_channels: int, out_layers):
        super().__init__(in_channels)
        backbone = MobileNet(backbone_mode=True, num_classes=None, in_channels=in_channels)
        self.multi_output_backbone = MultiOutputModule(backbone, out_layers)
        self._out_channels = [x.shape[1] for x in self.forward(torch.empty((1, in_channels, 64, 64)))]

    @property
    def out_channels(self) -> Union[List[int], int]:
        return self._out_channels

    def forward(self, x):
        return self.multi_output_backbone(x)


class MobileNetV2Backbone(BaseDetectionModule):
    """MobileNetV1 backbone with an option to return output of any layer"""

    def __init__(self, in_channels: int, width_mult, structure, grouped_conv_size, out_layers):
        super().__init__(in_channels)
        backbone = MobileNetV2(
            backbone_mode=True,
            num_classes=None,
            dropout=0.0,
            width_mult=width_mult,
            structure=structure,
            grouped_conv_size=grouped_conv_size,
            in_channels=in_channels,
        )
        self.multi_output_backbone = MultiOutputModule(backbone, out_layers)
        self._out_channels = [x.shape[1] for x in self.forward(torch.empty((1, in_channels, 64, 64)))]

    @property
    def out_channels(self) -> Union[List[int], int]:
        return self._out_channels

    def forward(self, x):
        return self.multi_output_backbone(x)


class SSDNeck(BaseDetectionModule, ABC):
    """
    SSD neck which returns:
     * outputs of the backbone, unchanged
     * outputs of a custom number of additional blocks added after the last backbone stage (returns output of each block)
    Has no skips to the backbone
    """

    def __init__(self, in_channels: Union[int, List[int]], blocks_out_channels, **kwargs):
        in_channels = in_channels if isinstance(in_channels, (list, ListConfig)) else [in_channels]
        super().__init__(in_channels)
        self.neck_blocks = nn.ModuleList(self.create_blocks(in_channels[-1], blocks_out_channels, **kwargs))
        self._out_channels = in_channels + list(blocks_out_channels)

    @property
    def out_channels(self) -> Union[List[int], int]:
        return self._out_channels

    @abstractmethod
    def create_blocks(self, in_channels: int, blocks_out_channels, **kwargs):
        raise NotImplementedError()

    def forward(self, inputs):
        outputs = inputs if isinstance(inputs, list) else [inputs]

        x = outputs[-1]
        for block in self.neck_blocks:
            x = block(x)
            outputs.append(x)

        return outputs


class SSDInvertedResidualNeck(SSDNeck):
    """
    Consecutive InvertedResidual blocks each starting with stride 2
    """

    def create_blocks(self, prev_channels, blocks_out_channels, expand_ratios, grouped_conv_size):
        neck_blocks = []
        for i in range(len(blocks_out_channels)):
            out_channels = blocks_out_channels[i]
            neck_blocks.append(InvertedResidual(prev_channels, out_channels, stride=2, expand_ratio=expand_ratios[i], grouped_conv_size=grouped_conv_size))
            prev_channels = out_channels
        return neck_blocks


class SSDBottleneckNeck(SSDNeck):
    """
    Consecutive bottleneck blocks
    """

    def create_blocks(self, prev_channels, blocks_out_channels, bottleneck_channels, kernel_sizes, strides):
        neck_blocks = []
        for i in range(len(blocks_out_channels)):
            mid_channels = bottleneck_channels[i]
            out_channels = blocks_out_channels[i]
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = 1 if stride == 2 else 0
            neck_blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, mid_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = out_channels
        return neck_blocks


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d."""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


class SSDHead(BaseDetectionModule):
    def __init__(self, in_channels: Union[int, List[int]], num_classes, anchors, lite):
        in_channels = in_channels if isinstance(in_channels, (list, ListConfig)) else [in_channels]
        super().__init__(in_channels)

        self.num_classes = num_classes
        self.dboxes_xy = nn.Parameter(anchors("xywh")[:, :2], requires_grad=False)
        self.dboxes_wh = nn.Parameter(anchors("xywh")[:, 2:], requires_grad=False)
        scale_xy = anchors.scale_xy
        scale_wh = anchors.scale_wh
        scales = torch.tensor([scale_xy, scale_xy, scale_wh, scale_wh])
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.img_size = nn.Parameter(torch.tensor([anchors.fig_size]), requires_grad=False)
        self.num_anchors = anchors.num_anchors

        loc_blocks = []
        conf_blocks = []

        for i, (num_anch, in_c) in enumerate(zip(self.num_anchors, in_channels)):
            conv = SeperableConv2d if lite and i < len(self.num_anchors) - 1 else nn.Conv2d
            loc_blocks.append(conv(in_c, num_anch * 4, kernel_size=3, padding=1))
            conf_blocks.append(conv(in_c, num_anch * (self.num_classes + 1), kernel_size=3, padding=1))

        self.loc = nn.ModuleList(loc_blocks)
        self.conf = nn.ModuleList(conf_blocks)

    @property
    def out_channels(self) -> Union[List[int], int]:
        return None

    def forward(self, inputs):
        inputs = inputs if isinstance(inputs, list) else [inputs]

        preds = []
        for i in range(len(inputs)):
            boxes_preds = self.loc[i](inputs[i])
            class_preds = self.conf[i](inputs[i])
            preds.append([boxes_preds, class_preds])

        return self.combine_preds(preds)

    def combine_preds(self, preds):
        batch_size = preds[0][0].shape[0]

        for i in range(len(preds)):
            box_pred_map, conf_pred_map = preds[i]
            preds[i][0] = box_pred_map.view(batch_size, 4, -1)
            preds[i][1] = conf_pred_map.view(batch_size, self.num_classes + 1, -1)

        locs, confs = list(zip(*preds))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()

        if self.training:
            # FOR 300X300 INPUT - RETURN N_BATCH X 8732 X {N_LABELS, N_LOCS} RESULTS
            return locs, confs
        else:
            bboxes_in = locs.permute(0, 2, 1)
            scores_in = confs.permute(0, 2, 1)

            bboxes_in *= self.scales

            # CONVERT RELATIVE LOCATIONS INTO ABSOLUTE LOCATION (OUTPUT LOCATIONS ARE RELATIVE TO THE DBOXES)
            xy = (bboxes_in[:, :, :2] * self.dboxes_wh + self.dboxes_xy) * self.img_size
            wh = (bboxes_in[:, :, 2:].exp() * self.dboxes_wh) * self.img_size

            # REPLACE THE CONFIDENCE OF CLASS NONE WITH OBJECT CONFIDENCE
            # SSD DOES NOT OUTPUT OBJECT CONFIDENCE, REQUIRED FOR THE NMS
            scores_in = torch.softmax(scores_in, dim=-1)
            classes_conf = scores_in[:, :, 1:]
            obj_conf = torch.max(classes_conf, dim=2)[0].unsqueeze(dim=-1)

            return torch.cat((xy, wh, obj_conf, classes_conf), dim=2), (locs, confs)


ALL_DETECTION_MODULES = {
    "MobileNetV1Backbone": MobileNetV1Backbone,
    "MobileNetV2Backbone": MobileNetV2Backbone,
    "SSDInvertedResidualNeck": SSDInvertedResidualNeck,
    "SSDBottleneckNeck": SSDBottleneckNeck,
    "SSDHead": SSDHead,
}
