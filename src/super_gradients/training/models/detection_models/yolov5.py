"""
YoloV5 code adapted from https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
"""
import math
from typing import Union, Type, List

import torch
import torch.nn as nn
from super_gradients.training.models.detection_models.csp_darknet53 import Conv, CSPDarknet53, get_yolo_version_params
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils.detection_utils import non_max_suppression, scale_img, \
    check_anchor_order, check_img_size_divisibilty, matrix_non_max_suppression, NMS_Type, \
    DetectionPostPredictionCallback, Anchors
from super_gradients.training.utils.export_utils import ExportableHardswish, ExportableSiLU
from super_gradients.training.utils.utils import HpmStruct, get_param

COCO_DETECTION_80_CLASSES_BBOX_ANCHORS = Anchors([[10, 13, 16, 30, 33, 23],
                                                  [30, 61, 62, 45, 59, 119],
                                                  [116, 90, 156, 198, 373, 326]],
                                                 strides=[8, 16, 32])  # output strides of all yolo outputs

DEFAULT_YOLOV5_ARCH_PARAMS = {
    'anchors': COCO_DETECTION_80_CLASSES_BBOX_ANCHORS,  # The sizes of the anchors predicted by the model
    'num_classes': 80,  # Number of classes to predict
    'depth_mult_factor': 1.0,  # depth multiplier for the entire model
    'width_mult_factor': 1.0,  # width multiplier for the entire model
    'channels_in': 3,  # # of classes the model predicts
    'skip_connections_list': [(12, [6]), (16, [4]), (19, [14]), (22, [10]), (24, [17, 20])],
    # A list defining skip connections. format is '[target: [source1, source2, ...]]'. Each item defines a skip
    # connection from all sources to the target according to the layer's index (count starts from the backbone)
    'connection_layers_input_channel_size': [1024, 1024, 512],
    # default number off channels for the connecting points between the backbone and the head
    'fuse_conv_and_bn': False,  # Fuse sequential Conv + B.N layers into a single one
    'add_nms': False,  # Add the NMS module to the computational graph
    'nms_conf': 0.25,  # When add_nms is True during NMS predictions with confidence lower than this will be discarded
    'nms_iou': 0.45,  # When add_nms is True IoU threshold for NMS algorithm
    # (with smaller value more boxed will be considered "the same" and removed)
    'yolo_version': 'v6.0'  # Release version of Ultralytics to built a model from: v.6.0 and v3.0 are supported
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

        self.m = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in channels)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.detection_layers_num):
            x[i] = self.m[i](x[i])  # conv
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
        self._skip_connections_dict = arch_params.skip_connections_dict
        # FLATTEN THE SOURCE LIST INTO A LIST OF INDICES
        self._layer_idx_to_extract = [idx for sub_l in self._skip_connections_dict.values() for idx in sub_l]

        # GET THREE CONNECTING POINTS CHANNEL INPUT SIZE
        connector = arch_params.connection_layers_input_channel_size

        _, block, activation_type, width_mult, depth_mult = get_yolo_version_params(arch_params.yolo_version,
                                                                                    arch_params.width_mult_factor,
                                                                                    arch_params.depth_mult_factor)

        self._modules_list = nn.ModuleList()
        self._modules_list.append(Conv(width_mult(connector[0]), width_mult(512), 1, 1, activation_type))  # 10
        self._modules_list.append(nn.Upsample(None, 2, 'nearest'))  # 11
        self._modules_list.append(Concat(1))  # 12
        self._modules_list.append(
            block(width_mult(connector[1]), width_mult(512), depth_mult(3), activation_type, False))  # 13

        self._modules_list.append(Conv(width_mult(512), width_mult(256), 1, 1, activation_type))  # 14
        self._modules_list.append(nn.Upsample(None, 2, 'nearest'))  # 15
        self._modules_list.append(Concat(1))  # 16
        self._modules_list.append(
            block(width_mult(connector[2]), width_mult(256), depth_mult(3), activation_type, False))  # 17

        self._modules_list.append(Conv(width_mult(256), width_mult(256), 3, 2, activation_type))  # 18
        self._modules_list.append(Concat(1))  # 19
        self._modules_list.append(block(width_mult(512), width_mult(512), depth_mult(3), activation_type, False))  # 20

        self._modules_list.append(Conv(width_mult(512), width_mult(512), 3, 2, activation_type))  # 21
        self._modules_list.append(Concat(1))  # 22
        self._modules_list.append(
            block(width_mult(1024), width_mult(1024), depth_mult(3), activation_type, False))  # 23

        self._modules_list.append(
            Detect(num_classes, anchors, channels=[width_mult(v) for v in (256, 512, 1024)]))  # 24

        self.width_mult = width_mult
        self.anchors = anchors

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
        self.arch_params = HpmStruct(**DEFAULT_YOLOV5_ARCH_PARAMS)
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
        check_anchor_order(m)

        self.register_buffer('stride', m.stride)  # USED ONLY FOR CONVERSION

    def _initialize_biases(self, cf=None):
        """initialize biases into Detect(), cf is class frequency"""
        # TODO: UNDERSTAND WHAT IS THIS cf AND IF WE NEED IT
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self._head._modules_list[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.num_anchors, -1)  # conv.bias(255) to (3,85)
            with torch.no_grad():
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.num_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.Hardswish]:
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
            res_flag, suggestion = check_img_size_divisibilty(dim, max_stride)
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


class Custom_YoLoV5(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        backbone = get_param(arch_params, 'backbone', YoLoV5DarknetBackbone)
        super().__init__(backbone=backbone, arch_params=arch_params)


class YoLoV5N(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.25
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)


class YoLoV5S(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.50
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)


class YoLoV5M(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.67
        arch_params.width_mult_factor = 0.75
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)


class YoLoV5L(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.0
        arch_params.width_mult_factor = 1.0
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)


class YoLoV5X(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.33
        arch_params.width_mult_factor = 1.25
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)
