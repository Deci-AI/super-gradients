import torch
import torch.nn as nn

from super_gradients.training.models import MobileNet, SgModule, MobileNetV2, InvertedResidual

from super_gradients.training.utils import HpmStruct, utils
from super_gradients.training.utils.module_utils import MultiOutputModule
from super_gradients.training.utils.ssd_utils import DefaultBoxes

DEFAULT_SSD_ARCH_PARAMS = {
    "additional_blocks_bottleneck_channels": [256, 256, 128, 128, 128]
}

DEFAULT_SSD_MOBILENET_V1_ARCH_PARAMS = {
    "out_channels": [512, 1024, 512, 256, 256, 256],
    "kernel_sizes": [3, 3, 3, 3, 2],
    "anchors": DefaultBoxes(fig_size=320, feat_size=[40, 20, 10, 5, 3, 2], scales=[22, 48, 106, 163, 221, 278, 336],
                            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], scale_xy=0.1, scale_wh=0.2)
}

DEFAULT_SSD_LITE_MOBILENET_V2_ARCH_PARAMS = {
    "out_channels": [576, 1280, 512, 256, 256, 64],
    "expand_ratios": [0.2, 0.25, 0.5, 0.25],
    "lite": True,
    "width_mult": 1.0,
    # "output_paths": [[7,'conv',2], [14, 'conv', 2]], output paths for a model with output levels of stride 8 plus
    "output_paths": [[14, 'conv', 2], 18],
    "anchors": DefaultBoxes(fig_size=320, feat_size=[20, 10, 5, 3, 2, 1], scales=[32, 82, 133, 184, 235, 285, 336],
                            aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]], scale_xy=0.1, scale_wh=0.2)
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

        # num classes in a dataset
        # the model will predict self.num_classes + 1 values to also include background
        self.num_classes = self.arch_params.num_classes
        self.dboxes_xy = nn.Parameter(self.arch_params.anchors('xywh')[:, :2], requires_grad=False)
        self.dboxes_wh = nn.Parameter(self.arch_params.anchors('xywh')[:, 2:], requires_grad=False)
        scale_xy = self.arch_params.anchors.scale_xy
        scale_wh = self.arch_params.anchors.scale_wh
        scales = torch.tensor([scale_xy, scale_xy, scale_wh, scale_wh])
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.img_size = nn.Parameter(torch.tensor([self.arch_params.anchors.fig_size]), requires_grad=False)
        self.num_anchors = self.arch_params.anchors.num_anchors

        self._build_additional_blocks()
        self._build_detecting_branches()
        self._init_weights()

    def _build_detecting_branches(self, build_loc=True):
        """Add localization and classification branches

        :param build_loc: whether to build localization branch;
                          called with False in replace_head(...), in such case only classification branch is rebuilt
        """
        if build_loc:
            self.loc = []
        self.conf = []

        out_channels = self.arch_params.out_channels
        lite = utils.get_param(self.arch_params, 'lite', False)
        for i, (nd, oc) in enumerate(zip(self.num_anchors, out_channels)):
            conv = SeperableConv2d if lite and i < len(self.num_anchors) - 1 else nn.Conv2d
            if build_loc:
                self.loc.append(conv(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(conv(oc, nd * (self.num_classes + 1), kernel_size=3, padding=1))

        if build_loc:
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

    def bbox_view(self, feature_maps):
        """ Shape the classifier to the view of bboxes """
        ret = []
        for features, loc, conf in zip(feature_maps, self.loc, self.conf):
            boxes_preds = loc(features).view(features.size(0), 4, -1)
            class_preds = conf(features).view(features.size(0), self.num_classes + 1, -1)
            ret.append((boxes_preds, class_preds))

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

        # detection_feed are FEATURE MAPS: i.e. FOR 300X300 INPUT - 38X38X4, 19X19X6, 10X10X6, 5X5X6, 3X3X4, 1X1X4
        locs, confs = self.bbox_view(detection_feed)

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

    def replace_head(self, new_num_classes):
        del self.conf
        self.arch_params.num_classes = new_num_classes
        self.num_classes = new_num_classes
        self._build_detecting_branches(build_loc=False)


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
        mobilenetv2 = MobileNetV2(num_classes=None, dropout=0.,
                                  backbone_mode=True, width_mult=self.arch_params.width_mult)
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
