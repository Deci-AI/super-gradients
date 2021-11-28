import torch
import torch.nn as nn

from super_gradients.training.models import MobileNet, SgModule, MobileNetV2, InvertedResidual

from super_gradients.training.utils import HpmStruct, utils
from super_gradients.training.utils.module_utils import MultiOutputModule

DEFAULT_SSD_ARCH_PARAMS = {
    "num_defaults": [4, 6, 6, 6, 4, 4],
    "additional_blocks_bottleneck_channels": [256, 256, 128, 128, 128]
}

DEFAULT_SSD_MOBILENET_V1_ARCH_PARAMS = {
    "out_channels": [512, 1024, 512, 256, 256, 256],
    "kernel_sizes": [3, 3, 3, 3, 2]
}

DEFAULT_SSD_LITE_MOBILENET_V2_ARCH_PARAMS = {
    "out_channels": [576, 1280, 512, 256, 256, 64],
    "expand_ratios": [0.2, 0.25, 0.5, 0.25],
    "num_defaults": [6, 6, 6, 6, 6, 6],
    "lite": True,
    "width_mult": 1.0,
    "output_paths": [[14, 'conv', 2], 18]
}


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                  groups=in_channels, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    )


class SSD(SgModule):
    """
    paper: https://arxiv.org/pdf/1512.02325.pdf
    based on code: https://github.com/NVIDIA/DeepLearningExamples
    """

    def __init__(self, backbone, arch_params):
        super().__init__()

        self.arch_params = HpmStruct(**DEFAULT_SSD_ARCH_PARAMS)
        self.arch_params.override(**arch_params.to_dict())

        paths = utils.get_param(self.arch_params, 'output_paths')
        if paths is not None:
            self.backbone = MultiOutputModule(backbone, paths)
        else:
            self.backbone = backbone

        lite = utils.get_param(arch_params, 'lite', False)
        # NUMBER OF CLASSES + 1 NO_CLASS
        self.num_classes = self.arch_params.num_classes

        self._build_additional_blocks()

        self._build_location_and_conf_branches(self.arch_params.out_channels, lite)

        self._init_weights()

    def _build_location_and_conf_branches(self, out_channels, lite: bool):
        """Add the sdd blocks after the backbone"""
        self.num_defaults = self.arch_params.num_defaults
        self.loc = []
        self.conf = []
        conv_to_use = SeperableConv2d if lite else nn.Conv2d
        for i, (nd, oc) in enumerate(zip(self.num_defaults, out_channels)):
            if i < len(self.num_defaults) - 1:
                self.loc.append(conv_to_use(oc, nd * 4, kernel_size=3, padding=1))
                self.conf.append(conv_to_use(oc, nd * self.num_classes, kernel_size=3, padding=1))
            else:
                self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
                self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))
        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)

    def _build_additional_blocks(self):
        input_size = self.arch_params.out_channels
        kernel_sizes = self.arch_params.kernel_sizes
        bottleneck_channels = self.arch_params.additional_blocks_bottleneck_channels

        self.additional_blocks = []
        for i, (input_size, output_size, channels, kernel_size) in enumerate(
                zip(input_size[:-1], input_size[1:], bottleneck_channels, kernel_sizes)):
            if i < 3:
                middle_layer = nn.Conv2d(channels, output_size, kernel_size=kernel_size, padding=1, stride=2,
                                         bias=False)
            else:
                middle_layer = nn.Conv2d(channels, output_size, kernel_size=kernel_size, bias=False)

            layer = nn.Sequential(
                nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                middle_layer,
                nn.BatchNorm2d(output_size),
                nn.ReLU(inplace=True),
            )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        """ Shape the classifier to the view of bboxes """
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.num_classes, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.backbone(x)

        # IF THE BACKBONE IS A MultiOutputModule WE GET A LIST, OTHERWISE WE WRAP IT IN A LIST
        detection_feed = x if isinstance(x, list) else [x]
        x = detection_feed[-1]

        for block in self.additional_blocks:
            x = block(x)
            detection_feed.append(x)

        # FEATURE MAPS: i.e. FOR 300X300 INPUT - 38X38X4, 19X19X6, 10X10X6, 5X5X6, 3X3X4, 1X1X4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # FOR 300X300 INPUT - RETURN N_BATCH X 8732 X {N_LABELS, N_LOCS} RESULTS
        return locs, confs


class SSDMobileNetV1(SSD):
    """
    paper: http://ceur-ws.org/Vol-2500/paper_5.pdf
    """

    def __init__(self, arch_params: HpmStruct):
        self.arch_params = HpmStruct(**DEFAULT_SSD_MOBILENET_V1_ARCH_PARAMS)
        self.arch_params.override(**arch_params.to_dict())
        mobilenet_backbone = MobileNet(num_classes=None, backbone_mode=True, up_to_layer=10)
        super().__init__(backbone=mobilenet_backbone, arch_params=self.arch_params)


class SSDLiteMobileNetV2(SSD):
    def __init__(self, arch_params: HpmStruct):
        self.arch_params = HpmStruct(**DEFAULT_SSD_LITE_MOBILENET_V2_ARCH_PARAMS)
        self.arch_params.override(**arch_params.to_dict())
        self.arch_params.out_channels[0] = int(round(self.arch_params.out_channels[0] * self.arch_params.width_mult))
        mobilenetv2 = MobileNetV2(num_classes=None, backbone_mode=True, width_mult=self.arch_params.width_mult)
        super().__init__(backbone=mobilenetv2.features, arch_params=self.arch_params)

    # OVERRIDE THE DEFAULT FUNCTION FROM SSD. ADD THE SDD BLOCKS AFTER THE BACKBONE.
    def _build_additional_blocks(self):
        channels = self.arch_params.out_channels
        expand_ratios = self.arch_params.expand_ratios
        self.additional_blocks = []
        for in_channels, out_channels, expand_ratio in zip(channels[1:-1], channels[2:], expand_ratios):
            self.additional_blocks.append(
                InvertedResidual(in_channels, out_channels, stride=2, expand_ratio=expand_ratio))

        self.additional_blocks = nn.ModuleList(self.additional_blocks)
