from functools import partial
from typing import Union, List, Type
from abc import abstractmethod, ABC

import torch
from torch import nn
from omegaconf.listconfig import ListConfig
from omegaconf import DictConfig

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.activations_type_factory import ActivationsTypeFactory
from super_gradients.modules import QARepVGGBlock

from super_gradients.common.registry.registry import register_detection_module
from super_gradients.training.models.classification_models.mobilenetv2 import InvertedResidual
from super_gradients.training.models.detection_models.csp_darknet53 import width_multiplier, Conv
from super_gradients.training.models.detection_models.deci_yolo.yolo_stages import CustomBlockCSPLayer
from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.models import MobileNet, MobileNetV2
from super_gradients.modules.multi_output_modules import MultiOutputModule


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
        """
        :return: channels of tensor(s) that will be returned by a module  in forward
        """
        raise NotImplementedError()


@register_detection_module()
class NStageBackbone(BaseDetectionModule):
    """
    A backbone with a stem -> N stages -> context module
    Returns outputs of the layers listed in out_layers
    """

    def __init__(
        self,
        in_channels: int,
        out_layers: List[str],
        stem: Union[str, HpmStruct, DictConfig],
        stages: Union[str, HpmStruct, DictConfig],
        context_module: Union[str, HpmStruct, DictConfig],
    ):
        import super_gradients.common.factories.detection_modules_factory as det_factory

        """
        :param out_layers: names of layers to output from the following options: 'stem', 'stageN', 'context_module'
        """
        super().__init__(in_channels)
        factory = det_factory.DetectionModulesFactory()

        self.num_stages = len(stages)
        self.stem = factory.get(factory.insert_module_param(stem, "in_channels", in_channels))
        prev_channels = self.stem.out_channels
        for i in range(self.num_stages):
            new_stage = factory.get(factory.insert_module_param(stages[i], "in_channels", prev_channels))
            setattr(self, f"stage{i + 1}", new_stage)
            prev_channels = new_stage.out_channels
        self.context_module = factory.get(factory.get(factory.insert_module_param(context_module, "in_channels", prev_channels)))

        self.out_layers = out_layers
        self._out_channels = self._define_out_channels()

    def _define_out_channels(self):
        out_channels = []
        for layer in self.out_layers:
            out_channels.append(getattr(self, layer).out_channels)
        return out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):

        outputs = []
        all_layers = ["stem"] + [f"stage{i}" for i in range(1, self.num_stages + 1)] + ["context_module"]
        for layer in all_layers:
            x = getattr(self, layer)(x)
            if layer in self.out_layers:
                outputs.append(x)

        return outputs


@register_detection_module()
class PANNeck(BaseDetectionModule):
    """
    A PAN (path aggregation network) neck with 4 stages (2 up-sampling and 2 down-sampling stages)
    Returns outputs of neck stage 2, stage 3, stage 4
    """

    def __init__(
        self,
        in_channels: List[int],
        neck1: Union[str, HpmStruct, DictConfig],
        neck2: Union[str, HpmStruct, DictConfig],
        neck3: Union[str, HpmStruct, DictConfig],
        neck4: Union[str, HpmStruct, DictConfig],
    ):
        import super_gradients.common.factories.detection_modules_factory as det_factory

        super().__init__(in_channels)
        c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        factory = det_factory.DetectionModulesFactory()
        self.neck1 = factory.get(factory.insert_module_param(neck1, "in_channels", [c5_out_channels, c4_out_channels]))
        self.neck2 = factory.get(factory.insert_module_param(neck2, "in_channels", [self.neck1.out_channels[1], c3_out_channels]))
        self.neck3 = factory.get(factory.insert_module_param(neck3, "in_channels", [self.neck2.out_channels[1], self.neck2.out_channels[0]]))
        self.neck4 = factory.get(factory.insert_module_param(neck4, "in_channels", [self.neck3.out_channels, self.neck1.out_channels[0]]))

        self._out_channels = [
            self.neck2.out_channels[1],
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        c3, c4, c5 = inputs

        x_n1_inter, x = self.neck1([c5, c4])
        x_n2_inter, p3 = self.neck2([x, c3])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5


@register_detection_module()
class NHeads(BaseDetectionModule):
    """
    Apply N heads in parallel and combine predictions into the shape expected by SG detection losses
    """

    def __init__(self, in_channels: List[int], num_classes: int, heads_list: Union[str, HpmStruct, DictConfig]):
        import super_gradients.common.factories.detection_modules_factory as det_factory

        super().__init__(in_channels)
        factory = det_factory.DetectionModulesFactory()
        heads_list = self._pass_num_classes(heads_list, factory, num_classes)

        self.num_heads = len(heads_list)
        for i in range(self.num_heads):
            new_head = factory.get(factory.insert_module_param(heads_list[i], "in_channels", in_channels[i]))
            setattr(self, f"head{i + 1}", new_head)

    @staticmethod
    def _pass_num_classes(heads_list, factory, num_classes):
        for i in range(len(heads_list)):
            heads_list[i] = factory.insert_module_param(heads_list[i], "num_classes", num_classes)
        return heads_list

    @property
    def out_channels(self):
        return None

    def forward(self, inputs):
        outputs = []
        for i in range(self.num_heads):
            outputs.append(getattr(self, f"head{i + 1}")(inputs[i]))

        return self.combine_preds(outputs)

    def combine_preds(self, preds):
        outputs = []
        outputs_logits = []
        for output, output_logits in preds:
            outputs.append(output)
            outputs_logits.append(output_logits)

        return outputs if self.training else (torch.cat(outputs, 1), outputs_logits)


class MultiOutputBackbone(BaseDetectionModule):
    """
    Defines a backbone using MultiOutputModule with the interface of BaseDetectionModule
    """

    def __init__(self, in_channels: int, backbone: nn.Module, out_layers: List):
        super().__init__(in_channels)
        self.multi_output_backbone = MultiOutputModule(backbone, out_layers)
        self._out_channels = [x.shape[1] for x in self.forward(torch.empty((1, in_channels, 64, 64)))]

    @property
    def out_channels(self) -> Union[List[int], int]:
        return self._out_channels

    def forward(self, x):
        return self.multi_output_backbone(x)


@register_detection_module()
class MobileNetV1Backbone(MultiOutputBackbone):
    """MobileNetV1 backbone with an option to return output of any layer"""

    def __init__(self, in_channels: int, out_layers: List):
        backbone = MobileNet(backbone_mode=True, num_classes=None, in_channels=in_channels)
        super().__init__(in_channels, backbone, out_layers)


@register_detection_module()
class MobileNetV2Backbone(MultiOutputBackbone):
    """MobileNetV2 backbone with an option to return output of any layer"""

    def __init__(self, in_channels: int, out_layers: List, width_mult: float = 1.0, structure: List[List] = None, grouped_conv_size: int = 1):
        backbone = MobileNetV2(
            backbone_mode=True,
            num_classes=None,
            dropout=0.0,
            width_mult=width_mult,
            structure=structure,
            grouped_conv_size=grouped_conv_size,
            in_channels=in_channels,
        )
        super().__init__(in_channels, backbone, out_layers)


class SSDNeck(BaseDetectionModule, ABC):
    """
    SSD neck which returns:
     * outputs of the backbone, unchanged
     * outputs of a custom number of additional blocks added after the last backbone stage (returns output of each block)
    Has no skips to the backbone
    """

    def __init__(self, in_channels: Union[int, List[int]], blocks_out_channels: List[int], **kwargs):
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


@register_detection_module()
class SSDInvertedResidualNeck(SSDNeck):
    """
    Consecutive InvertedResidual blocks each starting with stride 2
    """

    def create_blocks(self, prev_channels: int, blocks_out_channels: List[int], expand_ratios: List[float], grouped_conv_size: int):
        neck_blocks = []
        for i in range(len(blocks_out_channels)):
            out_channels = blocks_out_channels[i]
            neck_blocks.append(InvertedResidual(prev_channels, out_channels, stride=2, expand_ratio=expand_ratios[i], grouped_conv_size=grouped_conv_size))
            prev_channels = out_channels
        return neck_blocks


@register_detection_module()
class SSDBottleneckNeck(SSDNeck):
    """
    Consecutive bottleneck blocks
    """

    def create_blocks(self, prev_channels: int, blocks_out_channels: List[int], bottleneck_channels: List[int], kernel_sizes: List[int], strides: List[int]):
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


def SeperableConv2d(in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, bias: bool = True):
    """Depthwise Conv2d and Pointwise Conv2d."""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


@register_detection_module()
class SSDHead(BaseDetectionModule):
    """
    A one-layer conv head attached to each input feature map.
    A conv is implemented as two branches: localization and classification
    """

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


class DeciYOLOStem(BaseDetectionModule):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels)
        self._out_channels = out_channels
        self.conv = QARepVGGBlock(in_channels, out_channels, stride=2, use_residual_connection=False)

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.conv(x)


class DeciYOLOStage(BaseDetectionModule):
    @resolve_param("activation_type", ActivationsTypeFactory())
    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: int, activation_type: str, hidden_channels: int = None, concat_intermediates: bool = False
    ):
        super().__init__(in_channels)
        self._out_channels = out_channels
        self.downsample = QARepVGGBlock(in_channels, out_channels, stride=2, activation_type=activation_type, use_residual_connection=False)
        self.blocks = CustomBlockCSPLayer(
            out_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            True,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.blocks(self.downsample(x))


class UpDeciYOLOStage(BaseDetectionModule):
    @resolve_param("activation_type", ActivationsTypeFactory())
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: Type[nn.Module],
        hidden_channels: int = None,
        concat_intermediates: bool = False,
        reduce_channels: bool = False,
    ):
        super().__init__(in_channels)

        num_inputs = len(in_channels)
        if num_inputs == 2:
            in_channels, skip_in_channels = in_channels
        else:
            in_channels, skip_in_channels1, skip_in_channels2 = in_channels
            skip_in_channels = skip_in_channels1 + out_channels  # skip2 downsample results in out_channels channels
        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        if num_inputs == 2:
            self.reduce_skip = Conv(skip_in_channels, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
        else:
            self.reduce_skip1 = Conv(skip_in_channels1, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
            self.reduce_skip2 = Conv(skip_in_channels2, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()

        self.conv = Conv(in_channels, out_channels, 1, 1, activation_type)
        self.upsample = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)
        if num_inputs == 3:
            self.downsample = Conv(out_channels if reduce_channels else skip_in_channels2, out_channels, kernel=3, stride=2, activation_type=activation_type)

        self.reduce_after_concat = Conv(num_inputs * out_channels, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()

        after_concat_channels = out_channels if reduce_channels else out_channels + skip_in_channels
        self.blocks = CustomBlockCSPLayer(
            after_concat_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
        )

        self._out_channels = [out_channels, out_channels]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        if len(inputs) == 2:
            x, skip_x = inputs
            skip_x = [self.reduce_skip(skip_x)]
        else:
            x, skip_x1, skip_x2 = inputs
            skip_x1, skip_x2 = self.reduce_skip1(skip_x1), self.reduce_skip2(skip_x2)
            skip_x = [skip_x1, self.downsample(skip_x2)]
        x_inter = self.conv(x)
        x = self.upsample(x_inter)
        x = torch.cat([x, *skip_x], 1)
        x = self.reduce_after_concat(x)
        x = self.blocks(x)
        return x_inter, x


class DownDeciYOLOStage(BaseDetectionModule):
    @resolve_param("activation_type", ActivationsTypeFactory())
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: Type[nn.Module],
        hidden_channels: int = None,
        concat_intermediates: bool = False,
    ):
        super().__init__(in_channels)

        in_channels, skip_in_channels = in_channels
        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        self.conv = Conv(in_channels, out_channels // 2, 3, 2, activation_type)
        after_concat_channels = out_channels // 2 + skip_in_channels
        self.blocks = CustomBlockCSPLayer(
            after_concat_channels,
            out_channels,
            num_blocks,
            partial(Conv, kernel=3, stride=1),
            activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
        )

        self._out_channels = out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        x, skip_x = inputs
        x = self.conv(x)
        x = torch.cat([x, skip_x], 1)
        x = self.blocks(x)
        return x
