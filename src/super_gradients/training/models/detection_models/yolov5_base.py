"""
YoloV5 code adapted from https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
"""
import math
from typing import Union, Type, List

import torch
import torch.nn as nn
from super_gradients.training.models.detection_models.csp_darknet53 import Conv, GroupedConvBlock, \
    CSPDarknet53, get_yolo_version_params
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils.detection_utils import non_max_suppression, scale_img, \
    check_anchor_order, matrix_non_max_suppression, NMS_Type, DetectionPostPredictionCallback, Anchors
from super_gradients.training.utils.export_utils import ExportableHardswish, ExportableSiLU
from super_gradients.training.utils.utils import HpmStruct, check_img_size_divisibility, get_param


COCO_DETECTION_80_CLASSES_BBOX_ANCHORS = Anchors([[10, 13, 16, 30, 33, 23],
                                                  [30, 61, 62, 45, 59, 119],
                                                  [116, 90, 156, 198, 373, 326]],
                                                 strides=[8, 16, 32])  # output strides of all yolo outputs

ANCHORSLESS_DUMMY_ANCHORS = Anchors([[], [], []], strides=[8, 16, 32])


DEFAULT_YOLO_ARCH_PARAMS = {
    'num_classes': 80,  # Number of classes to predict
    'depth_mult_factor': 1.0,  # depth multiplier for the entire model
    'width_mult_factor': 1.0,  # width multiplier for the entire model
    'channels_in': 3,  # Number of channels in the input image
    'skip_connections_list': [(12, [6]), (16, [4]), (19, [14]), (22, [10]), (24, [17, 20])],
    # A list defining skip connections. format is '[target: [source1, source2, ...]]'. Each item defines a skip
    # connection from all sources to the target according to the layer's index (count starts from the backbone)
    'backbone_connection_channels': [1024, 512, 256],  # width of backbone channels that are concatenated with the head
    # True if width_mult_factor is applied to the backbone (is the case with the default backbones)
    # which means that backbone_connection_channels should be used with a width_mult_factor
    # False if backbone_connection_channels should be used as is
    'scaled_backbone_width': True,
    'fuse_conv_and_bn': False,  # Fuse sequential Conv + B.N layers into a single one
    'add_nms': False,  # Add the NMS module to the computational graph
    'nms_conf': 0.25,  # When add_nms is True during NMS predictions with confidence lower than this will be discarded
    'nms_iou': 0.45,  # When add_nms is True IoU threshold for NMS algorithm
    # (with smaller value more boxed will be considered "the same" and removed)
    'yolo_type': 'yoloV5',  # Type of yolo to build: 'yoloV5' and 'yoloX' are supported
    'yolo_version': 'v6.0',  # Release version of Ultralytics yoloV5 to build a model from: v6.0 and v3.0 are supported
                             # (has an impact only if yolo_type is yoloV5)
    'stem_type': None,  # 'focus' and '6x6' are supported, by default is defined by yolo_type and yolo_version
    'depthwise': False,  # use depthwise separable convolutions all over the model
    'xhead_inter_channels': None,  # (has an impact only if yolo_type is yoloX)
    # Channels in classification and regression branches of the detecting blocks;
    # if is None the first of input channels will be used by default
    'xhead_groups': None,  # (has an impact only if yolo_type is yoloX)
    # Num groups in convs in classification and regression branches of the detecting blocks;
    # if None default groups will be used according to conv type
    # (1 for Conv and depthwise for GroupedConvBlock)
}


class YoloV5PostPredictionCallback(DetectionPostPredictionCallback):
    """Non-Maximum Suppression (NMS) module"""

    def __init__(self, conf: float = 0.001, iou: float = 0.6, classes: List[int] = None,
                 nms_type: NMS_Type = NMS_Type.ITERATIVE, max_predictions: int = 300):
        """
        :param conf: confidence threshold
        :param iou: IoU threshold                                       (used in NMS_Type.ITERATIVE)
        :param classes: (optional list) filter by class                 (used in NMS_Type.ITERATIVE)
        :param nms_type: the type of nms to use (iterative or matrix)
        :param max_predictions: maximum number of boxes to output       (used in NMS_Type.MATRIX)
        """
        super(YoloV5PostPredictionCallback, self).__init__()
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.nms_type = nms_type
        self.max_predictions = max_predictions

    def forward(self, x, device: str = None):
        if self.nms_type == NMS_Type.ITERATIVE:
            return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
        else:
            return matrix_non_max_suppression(x[0], conf_thres=self.conf, max_num_of_detections=self.max_predictions)


class Concat(nn.Module):
    """ CONCATENATE A LIST OF TENSORS ALONG DIMENSION"""

    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, self.dimension)


class Detect(nn.Module):

    def __init__(self, num_classes: int, anchors: Anchors, channels: list = None):
        super().__init__()

        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.detection_layers_num = anchors.detection_layers_num
        self.num_anchors = anchors.num_anchors
        self.grid = [torch.zeros(1)] * self.detection_layers_num  # init grid

        self.register_buffer('stride', anchors.stride)
        self.register_buffer('anchors', anchors.anchors)
        self.register_buffer('anchor_grid', anchors.anchor_grid)

        self.output_convs = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in channels)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.detection_layers_num):
            x[i] = self.output_convs[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.num_anchors, 1, 1, 2)  # wh
                y = torch.cat([xy, wh, y[..., 4:]], dim=4)
                z.append(y.view(bs, -1, self.num_outputs))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class DetectX(nn.Module):

    def __init__(self, num_classes: int, stride: torch.Tensor, activation_func_type: type, channels: list,
                 depthwise=False, groups: int = None, inter_channels: Union[int, List] = None):
        """
        :param stride:          strides of each predicting level
        :param channels:        input channels into all detecting layers
                                (from all neck layers that will be used for predicting)
        :param depthwise:       defines conv type in classification and regression branches (Conv or GroupedConvBlock)
                                depthwise is False by default in favor of a usual Conv
        :param groups:          num groups in convs in classification and regression branches;
                                if None default groups will be used according to conv type
                                (1 for Conv and depthwise for GroupedConvBlock)
        :param inter_channels:  channels in classification and regression branches;
                                if None channels[0] will be used by default
        """
        super().__init__()

        self.num_classes = num_classes
        self.detection_layers_num = len(channels)
        self.n_anchors = 1
        self.grid = [torch.zeros(1)] * self.detection_layers_num  # init grid

        self.register_buffer('stride', stride)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        ConvBlock = GroupedConvBlock if depthwise else Conv

        inter_channels = inter_channels or channels[0]
        inter_channels = inter_channels if isinstance(inter_channels, list) \
            else [inter_channels] * self.detection_layers_num
        for i in range(self.detection_layers_num):
            self.stems.append(Conv(channels[i], inter_channels[i], 1, 1, activation_func_type))

            self.cls_convs.append(
                nn.Sequential(*[ConvBlock(inter_channels[i], inter_channels[i], 3, 1, activation_func_type,
                                          groups=groups),
                                ConvBlock(inter_channels[i], inter_channels[i], 3, 1, activation_func_type,
                                          groups=groups)]))
            self.reg_convs.append(
                nn.Sequential(*[ConvBlock(inter_channels[i], inter_channels[i], 3, 1, activation_func_type,
                                          groups=groups),
                                ConvBlock(inter_channels[i], inter_channels[i], 3, 1, activation_func_type,
                                          groups=groups)]))

            self.cls_preds.append(nn.Conv2d(inter_channels[i], self.n_anchors * self.num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(inter_channels[i], 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(inter_channels[i], self.n_anchors * 1, 1, 1, 0))

    def forward(self, inputs):
        outputs = []
        outputs_logits = []
        for i in range(self.detection_layers_num):
            x = self.stems[i](inputs[i])

            cls_feat = self.cls_convs[i](x)
            cls_output = self.cls_preds[i](cls_feat)

            reg_feat = self.reg_convs[i](x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            bs, _, ny, nx = reg_feat.shape
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output = output.view(bs, self.n_anchors, -1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                outputs_logits.append(output.clone())
                if self.grid[i].shape[2:4] != output.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(output.device)

                xy = (output[..., :2] + self.grid[i].to(output.device)) * self.stride[i]
                wh = torch.exp(output[..., 2:4]) * self.stride[i]
                output = torch.cat([xy, wh, output[..., 4:].sigmoid()], dim=4)
                output = output.view(bs, -1, output.shape[-1])

            outputs.append(output)

        return outputs if self.training else (torch.cat(outputs, 1), outputs_logits)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class AbstractYoLoV5Backbone:
    def __init__(self, arch_params):
        # CREATE A LIST CONTAINING THE LAYERS TO EXTRACT FROM THE BACKBONE AND ADD THE FINAL LAYER
        self._layer_idx_to_extract = [idx for sub_l in arch_params.skip_connections_dict.values() for idx in sub_l]
        self._layer_idx_to_extract.append(len(self._modules_list) - 1)

    def forward(self, x):
        """:return A list, the length of self._modules_list containing the output of the layer if specified in
        self._layers_to_extract and None otherwise"""
        extracted_intermediate_layers = []
        for layer_idx, layer_module in enumerate(self._modules_list):
            # PREDICT THE NEXT LAYER'S OUTPUT
            x = layer_module(x)
            # IF INDICATED APPEND THE OUTPUT TO extracted_intermediate_layers O.W. APPEND None
            extracted_intermediate_layers.append(x) if layer_idx in self._layer_idx_to_extract \
                else extracted_intermediate_layers.append(None)

        return extracted_intermediate_layers


class YoLoV5DarknetBackbone(AbstractYoLoV5Backbone, CSPDarknet53):
    """Implements the CSP_Darknet53 module and inherit the forward pass to extract layers indicated in arch_params"""

    def __init__(self, arch_params):
        arch_params.backbone_mode = True
        CSPDarknet53.__init__(self, arch_params)
        AbstractYoLoV5Backbone.__init__(self, arch_params)

    def forward(self, x):
        return AbstractYoLoV5Backbone.forward(self, x)


class YoLoV5Head(nn.Module):
    def __init__(self, arch_params):
        super().__init__()
        # PARSE arch_params
        num_classes = arch_params.num_classes
        anchors = arch_params.anchors
        depthwise = arch_params.depthwise
        xhead_groups = arch_params.xhead_groups
        xhead_inter_channels = arch_params.xhead_inter_channels

        self._skip_connections_dict = arch_params.skip_connections_dict
        # FLATTEN THE SOURCE LIST INTO A LIST OF INDICES
        self._layer_idx_to_extract = [idx for sub_l in self._skip_connections_dict.values() for idx in sub_l]

        _, block, activation_type, width_mult, depth_mult = get_yolo_version_params(arch_params.yolo_version,
                                                                                    arch_params.yolo_type,
                                                                                    arch_params.width_mult_factor,
                                                                                    arch_params.depth_mult_factor)

        backbone_connector = [width_mult(c) if arch_params.scaled_backbone_width else c
                             for c in arch_params.backbone_connection_channels]

        DownConv = GroupedConvBlock if depthwise else Conv

        self._modules_list = nn.ModuleList()
        self._modules_list.append(Conv(backbone_connector[0], width_mult(512), 1, 1, activation_type))  # 10
        self._modules_list.append(nn.Upsample(None, 2, 'nearest'))  # 11
        self._modules_list.append(Concat(1))  # 12
        self._modules_list.append(
            block(backbone_connector[1] + width_mult(512), width_mult(512), depth_mult(3), activation_type, False,
                  depthwise))  # 13

        self._modules_list.append(Conv(width_mult(512), width_mult(256), 1, 1, activation_type))  # 14
        self._modules_list.append(nn.Upsample(None, 2, 'nearest'))  # 15
        self._modules_list.append(Concat(1))  # 16
        self._modules_list.append(
            block(backbone_connector[2] + width_mult(256), width_mult(256), depth_mult(3), activation_type, False,
                  depthwise))  # 17

        self._modules_list.append(DownConv(width_mult(256), width_mult(256), 3, 2, activation_type))  # 18
        self._modules_list.append(Concat(1))  # 19
        self._modules_list.append(
            block(2 * width_mult(256), width_mult(512), depth_mult(3), activation_type, False, depthwise))  # 20

        self._modules_list.append(DownConv(width_mult(512), width_mult(512), 3, 2, activation_type))  # 21
        self._modules_list.append(Concat(1))  # 22
        self._modules_list.append(
            block(2 * width_mult(512), width_mult(1024), depth_mult(3), activation_type, False, depthwise))  # 23

        detect_input_channels = [width_mult(v) for v in (256, 512, 1024)]
        if arch_params.yolo_type != 'yoloX':
            self._modules_list.append(
                Detect(num_classes, anchors, channels=detect_input_channels))  # 24
        else:
            strides = anchors.stride
            self._modules_list.append(
                DetectX(num_classes, strides, activation_type, channels=detect_input_channels, depthwise=depthwise,
                        groups=xhead_groups, inter_channels=xhead_inter_channels))  # 24

        self.anchors = anchors
        self.width_mult = width_mult

    def forward(self, intermediate_output):
        """
        :param intermediate_output: A list of the intermediate prediction of layers specified in the
        self._inter_layer_idx_to_extract from the Backbone
        """
        # COUNT THE NUMBER OF LAYERS IN THE BACKBONE TO CONTINUE THE COUNTER
        num_layers_in_backbone = len(intermediate_output)
        # INPUT TO HEAD IS THE LAST ELEMENT OF THE BACKBONE'S OUTPUT
        out = intermediate_output[-1]
        # RUN OVER THE MODULE LIST WITHOUT THE FINAL LAYER & START COUNTER FROM THE END OF THE BACKBONE
        for layer_idx, layer_module in enumerate(self._modules_list[:-1], start=num_layers_in_backbone):
            # IF THE LAYER APPEARS IN THE KEYS IT INSERT THE PRECIOUS OUTPUT AND THE INDICATED SKIP CONNECTIONS
            out = layer_module([out, intermediate_output[self._skip_connections_dict[layer_idx][0]]]) \
                if layer_idx in self._skip_connections_dict.keys() else layer_module(out)

            # IF INDICATED APPEND THE OUTPUT TO inter_layer_idx_to_extract O.W. APPEND None
            intermediate_output.append(out) if layer_idx in self._layer_idx_to_extract \
                else intermediate_output.append(None)

        # INSERT THE REMAINING LAYERS INTO THE Detect LAYER
        last_idx = len(self._modules_list) + num_layers_in_backbone - 1
        return self._modules_list[-1]([intermediate_output[self._skip_connections_dict[last_idx][0]],
                                       intermediate_output[self._skip_connections_dict[last_idx][1]],
                                       out])


class YoLoV5Base(SgModule):
    def __init__(self, backbone: Type[nn.Module], arch_params: HpmStruct, initialize_module: bool = True):
        super().__init__()
        # DEFAULT PARAMETERS TO BE OVERWRITTEN BY DUPLICATES THAT APPEAR IN arch_params
        self.arch_params = HpmStruct(**DEFAULT_YOLO_ARCH_PARAMS)
        if get_param(arch_params, 'yolo_type', 'yoloV5') != 'yoloX':
            self.arch_params.anchors = COCO_DETECTION_80_CLASSES_BBOX_ANCHORS
        else:
            self.arch_params.anchors = ANCHORSLESS_DUMMY_ANCHORS
        self.arch_params.override(**arch_params.to_dict())
        self.arch_params.skip_connections_dict = {k: v for k, v in self.arch_params.skip_connections_list}

        self.num_classes = self.arch_params.num_classes
        # THE MODEL'S MODULES
        self._backbone = backbone(arch_params=self.arch_params)
        self._nms = nn.Identity()

        # A FLAG TO DEFINE augment_forward IN INFERENCE
        self.augmented_inference = False

        # RUN SPECIFIC INITIALIZATION OF YOLO-V5
        if initialize_module:
            self._head = YoLoV5Head(self.arch_params)
            self._initialize_module()

    def forward(self, x):
        return self._augment_forward(x) if self.augmented_inference else self._forward_once(x)

    def _forward_once(self, x):
        out = self._backbone(x)
        out = self._head(out)
        # THIS HAS NO EFFECT IF add_nms() WAS NOT DONE
        out = self._nms(out)
        return out

    def _augment_forward(self, x):
        """Multi-scale forward pass"""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si)
            yi = self._forward_once(xi)[0]  # forward
            yi[..., :4] /= si  # de-scale
            if fi == 2:
                yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
            elif fi == 3:
                yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def load_state_dict(self, state_dict, strict=True):
        try:
            super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            raise RuntimeError(f"Got exception {e}, if a mismatch between expected and given state_dict keys exist, "
                               f"checkpoint may have been saved after fusing conv and bn. use fuse_conv_bn before loading.")

    def _initialize_module(self):
        self._check_strides_and_anchors()
        self._initialize_biases()
        self._initialize_weights()
        if self.arch_params.add_nms:
            nms_conf = self.arch_params.nms_conf
            nms_iou = self.arch_params.nms_iou
            self._nms = YoloV5PostPredictionCallback(nms_conf, nms_iou)

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int,
                            training_params: HpmStruct, total_batch: int) -> list:

        lr_warmup_epochs = get_param(training_params, 'lr_warmup_epochs', 0)
        if epoch >= lr_warmup_epochs:
            return super().update_param_groups(param_groups, lr, epoch, iter, training_params, total_batch)
        else:
            return param_groups

    def _check_strides_and_anchors(self):
        m = self._head._modules_list[-1]  # Detect()
        # Do inference in train mode on a dummy image to get output stride of each head output layer
        s = 128  # twice the minimum acceptable image size
        dummy_input = torch.zeros(1, self.arch_params.channels_in, s, s)
        dummy_input = dummy_input.to(next(self._backbone.parameters()).device)
        stride = torch.tensor([s / x.shape[-2] for x in self._forward_once(dummy_input)])
        stride = stride.to(dummy_input.device)
        if not torch.equal(m.stride, stride):
            raise RuntimeError('Provided anchor strides do not match the model strides')
        if isinstance(m, Detect):
            check_anchor_order(m)

        self.register_buffer('stride', m.stride)  # USED ONLY FOR CONVERSION

    def _initialize_biases(self, cf=None):
        """initialize biases into Detect(), cf is class frequency"""
        detect_module = self._head._modules_list[-1]  # Detect() module
        if isinstance(detect_module, Detect):
            for pred_conv, s in zip(detect_module.output_convs, detect_module.stride):  # from
                bias = pred_conv.bias.view(detect_module.num_anchors, -1)  # conv.bias(255) to (3,85)
                with torch.no_grad():
                    bias[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                    bias[:, 5:] += math.log(0.6 / (detect_module.num_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                pred_conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
        elif isinstance(detect_module, DetectX):
            prior_prob = 1e-2
            for conv in detect_module.cls_preds:
                bias = conv.bias.view(detect_module.n_anchors, -1)
                bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

            for conv in detect_module.obj_preds:
                bias = conv.bias.view(detect_module.n_anchors, -1)
                bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU)):
                m.inplace = True

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        initialize_optimizer_for_model_param_groups - Initializes the weights of the optimizer
                                                      adds weight decay  *Only* to the Conv2D layers
            :param optimizer_cls:   The nn.optim (optimizer class) to initialize
            :param lr:              lr to set for the optimizer
            :param training_params:
            :return: The optimizer, initialized with the relevant param groups
        """
        optimizer_params = get_param(training_params, 'optimizer_params')
        # OPTIMIZER PARAMETER GROUPS
        default_param_group, weight_decay_param_group, biases_param_group = [], [], []
        deprecated_params_total = 0

        for name, m in self.named_modules():
            if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):  # bias
                biases_param_group.append((name, m.bias))
            if isinstance(m, nn.BatchNorm2d):  # weight (no decay)
                default_param_group.append((name, m.weight))
            elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):  # weight (with decay)
                weight_decay_param_group.append((name, m.weight))
            elif name == '_head.anchors':
                deprecated_params_total += m.stride.numel() + m._anchors.numel() + m._anchor_grid.numel()

        # EXTRACT weight_decay FROM THE optimizer_params IN ORDER TO ASSIGN THEM MANUALLY
        weight_decay = optimizer_params.pop('weight_decay') if 'weight_decay' in optimizer_params.keys() else 0
        param_groups = [{'named_params': default_param_group, 'lr': lr, **optimizer_params, 'name': 'default'},
                        {'named_params': weight_decay_param_group, 'weight_decay': weight_decay, 'name': 'wd'},
                        {'named_params': biases_param_group, 'name': 'bias'}]

        # Assert that all parameters were added to optimizer param groups
        params_total = sum(p.numel() for p in self.parameters())
        optimizer_params_total = sum(p.numel() for g in param_groups for _, p in g['named_params'])
        assert params_total == optimizer_params_total + deprecated_params_total, \
            f"Parameters {[n for n, _ in self.named_parameters() if 'weight' not in n and 'bias' not in n]} " \
            f"weren't added to optimizer param groups"

        return param_groups

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        A method for preparing the YoloV5 model for conversion to other frameworks (ONNX, CoreML etc)
        :param input_size: expected input size
        :return:
        """
        assert not self.training, 'model has to be in eval mode to be converted'

        # Verify dummy_input from converter is of multiple of the grid size
        max_stride = int(max(self.stride))

        # Validate the image size
        image_dims = input_size[-2:]  # assume torch uses channels first layout
        for dim in image_dims:
            res_flag, suggestion = check_img_size_divisibility(dim, max_stride)
            if not res_flag:
                raise ValueError(f'Invalid input size: {input_size}. The input size must be multiple of max stride: '
                                 f'{max_stride}. The closest suggestions are: {suggestion[0]}x{suggestion[0]} or '
                                 f'{suggestion[1]}x{suggestion[1]}')

        # Update the model with exportable operators
        for k, m in self.named_modules():
            if isinstance(m, Conv):
                if isinstance(m.act, nn.Hardswish):
                    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
                    m.act = ExportableHardswish()  # assign activation
                elif isinstance(m.act, nn.SiLU):
                    m.act = ExportableSiLU()  # assign activation

    def get_include_attributes(self) -> list:
        return ["grid", "anchors", "anchors_grid"]

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self._head = new_head
        else:
            self.arch_params.num_classes = new_num_classes
            new_last_layer = Detect(new_num_classes, self._head.anchors, channels=[self._head.width_mult(v) for v in (256, 512, 1024)])
            new_last_layer = new_last_layer.to(next(self.parameters()).device)
            self._head._modules_list[-1] = new_last_layer
            self._check_strides_and_anchors()
            self._initialize_biases()
            self._initialize_weights()
