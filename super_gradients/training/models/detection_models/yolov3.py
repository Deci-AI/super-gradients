"""
Yolov3 code adapted from https://github.com/ultralytics/yolov3
"""
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from super_gradients.training.models import SgModule
from super_gradients.training.models.detection_models.darknet53 import Darknet53, DarkResidualBlock, create_conv_module
from super_gradients.training.utils import HpmStruct, get_param


class SPPLayer(nn.Module):
    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_4 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        return out


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOLayer(nn.Module):
    def __init__(self, anchors_mask: list, classes_num: int, anchors: list, image_size: int, onnx_stride: int,
                 onnx_export_mode: bool = False):
        """
        YOLOLayer
            :param anchors_mask:
            :param classes_num:
            :param anchors:
            :param image_size:
            :param onnx_stride:
            :param onnx_export_mode:
        """
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.onnx_export_mode = onnx_export_mode
        masked_anchors = [self.anchors[i] for i in anchors_mask]
        anchors = np.array(masked_anchors)
        self.anchors_mask = torch.Tensor(anchors)

        self.anchors_num = len(anchors_mask)
        self.classes_num = classes_num
        self.x_grid_points_num = 0
        self.y_grid_points_num = 0
        self.onnx_stride = onnx_stride

    def forward(self, img, img_size):
        if self.onnx_export_mode:
            # ALL OF THE GRIDS WERE CALCULATED IN init
            batch_size = 1
        else:
            batch_size, _, y_grid_points_num, x_grid_points_num = img.shape
            if (self.x_grid_points_num, self.y_grid_points_num) != (x_grid_points_num, y_grid_points_num):
                self.create_grids(img_size, (x_grid_points_num, y_grid_points_num), img.device, img.dtype)

        # PREDICTION
        # IMG.VIEW(BATCH_SIZE, PRE_YOLO_LAYER_SIZE(DEFAULT IS 255), 13, 13) --> (BATCH_SIZE, 3, 13, 13, NUM_CLASSES + 5)
        # (BS, ANCHORS_NUM, GRID, GRID, CLASSES + XYWH + OBJECTNESS)
        prediction = img.view(batch_size, self.anchors_num, self.classes_num + 5, self.y_grid_points_num,
                              self.x_grid_points_num).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return prediction

        # INFERENCE - ONNX
        elif self.onnx_export_mode:
            # CONSTANTS CAN NOT BE BROADCASTED
            m = self.anchors_num * self.x_grid_points_num * self.y_grid_points_num
            ngu = self.grid_size.repeat((1, m, 1))
            grid_xy = self.grid_xy.repeat((1, self.anchors_num, 1, 1, 1)).view(1, m, 2)
            anchor_wh = self.anchor_wh.repeat((1, 1, self.x_grid_points_num, self.y_grid_points_num, 1)).view(1, m,
                                                                                                              2) / ngu

            # MOVE THE TENSORS TO SAME DEVICE AS prediction TO APPLY TENSOR CALCULATION
            ngu = ngu.to(prediction.device)
            grid_xy = grid_xy.to(prediction.device)
            anchor_wh = anchor_wh.to(prediction.device)

            prediction = prediction.view(m, 5 + self.classes_num)
            xy = torch.sigmoid(prediction[..., 0:2]) + grid_xy[0]  # x, y
            wh = torch.exp(prediction[..., 2:4]) * anchor_wh[0]  # width, height
            prediction_confidence = torch.sigmoid(prediction[:, 4:5])

            # CHANGE THE RESULTS TO BE A VECTOR OF CLASS CONF * OBJECTNESS CONF FOR EACH OF THE CLASSES (like SSD)
            cls_prediction = F.softmax(prediction[:, 5:5 + self.classes_num], 1) * prediction_confidence
            return torch.cat((xy / ngu[0], wh, prediction_confidence, cls_prediction), 1).t()

        # INFERENCE
        else:
            inference_out = prediction.clone()
            inference_out[..., 0:2] = torch.sigmoid(inference_out[..., 0:2]) + self.grid_xy
            inference_out[..., 2:4] = torch.exp(inference_out[..., 2:4]) * self.anchor_wh
            inference_out[..., :4] *= self.stride
            torch.sigmoid_(inference_out[..., 4:])

            if self.classes_num == 1:
                # IGNORE cls FOR SINGLE CLASS DATA SETS
                inference_out[..., 5] = 1

            # RESHAPE FROM [1, 3, 13, 13, NUM_CLASSES + 5] TO [1, 507, NUM_CLASSES + 5]
            return inference_out.view(batch_size, -1, 5 + self.classes_num), prediction

    def create_grids(self, img_size=(416, 416), grid_size=(13, 13), device='cpu', data_type=torch.float32):
        """
        create_grids - Creates the grids for image sizes that are different than the model's defualt image size
            :param img_size:
            :param grid_size:
            :param device:
            :param data_type:
        """
        nx, ny = grid_size
        self.img_size = max(img_size)
        self.stride = self.img_size / max(grid_size)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid_xy = torch.stack((xv, yv), 2).to(device).type(data_type).view((1, 1, ny, nx, 2))

        # build wh gains
        self.anchor_vec = self.anchors_mask.to(device) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.anchors_num, 1, 1, 2).to(device).type(data_type)
        self.grid_size = torch.Tensor(grid_size).to(device)
        self.x_grid_points_num = nx
        self.y_grid_points_num = ny


class YoloV3(SgModule):
    """
    YoloV3
    """

    def __init__(self, num_classes: int = 80, image_size: int = 416,
                 arch_params: HpmStruct = None,
                 iou_t: float = 0.225, yolo_v3_anchors: list = None, onnx_export_mode=False):
        super(YoloV3, self).__init__()

        if arch_params:
            arch_params_dict = arch_params.to_dict()
            self.num_classes = arch_params.num_classes if 'num_classes' in arch_params_dict else num_classes
            self.image_size = arch_params.image_size if 'image_size' in arch_params_dict else image_size
            self.iou_t = arch_params.iou_t if 'iou_t' in arch_params_dict else iou_t
            self.onnx_export_mode = arch_params.onnx_export_mode if \
                'onnx_export_mode' in arch_params_dict else onnx_export_mode
            yolo_v3_anchors = arch_params.yolo_v3_anchors if 'yolo_v3_anchors' in arch_params_dict else yolo_v3_anchors

        else:
            self.image_size = image_size
            self.num_classes = num_classes
            self.iou_t = iou_t
            self.onnx_export_mode = onnx_export_mode

        # THIS IS THE LAYER SIZE THAT FEEDS THE YOLO LAYER
        self.pre_yolo_layer_size = (self.num_classes + 5) * 3

        if yolo_v3_anchors is None:
            # USE DEFAULT COCO DATA SET ANCHORS FOR YOLO V3
            yolo_v3_anchors = [
                (10., 13.), (16., 30.), (33., 23.),
                (30., 61.), (62., 45.), (59., 119.),
                (116., 90.), (156., 198.), (373., 326.)]

        self.yolo_v3_anchors = yolo_v3_anchors
        self.module_list = self.create_modules_list(num_classes=self.num_classes)
        self.yolo_layers_indices = self.get_yolo_layers_indices()

        if self.onnx_export_mode:
            self.prep_model_for_conversion([self.image_size, self.image_size])

    def forward(self, x, var=None):
        img_size = x.shape[-2:]
        yolo_output = []
        route_layers = []

        for i, module in enumerate(self.module_list):

            if isinstance(module, YOLOLayer):
                y = module(x, img_size=img_size)
                yolo_output.append(y)

            else:
                x = module(x)

            # CONCATENATE THE OUTPUTS OF PREVIOUS LAYERS
            x, route_layers = self.concatenate_layer_output(x, i, route_layers)

        if self.training:
            return yolo_output

        elif self.onnx_export_mode:
            #  CAT 3 LAYERS (NUM_CLASSES + 5) X (507, 2028, 8112) TO (NUM_CLASSES + 5) X 10647
            output = torch.cat(yolo_output, 1)
            # ONNX SCORES, bboxes
            return output[5:5 + self.num_classes].t(), output[0:4].t()

        else:
            # INFERENCE
            inference_output, training_output = list(zip(*yolo_output))
            return torch.cat(inference_output, 1), training_output

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        initialize_optimizer_for_model_param_groups - Initializes the optimizer group params,
                                                      adds weight decay  *Only* to the Conv2D layers
            :param lr:              lr to set for the optimizer
            :param training_params:
            :return:  A dictionary with named params and optimizer attributes
        """
        optimizer_params = get_param(training_params, 'optimizer_params')
        # OPTIMIZER PARAMETER GROUPS
        default_param_group, weight_decay_param_group, biases_param_group = [], [], []

        for k, v in dict(self.named_parameters()).items():
            if '.bias' in k:
                biases_param_group += [[k, v]]
            elif 'Conv2d.weight' in k:
                weight_decay_param_group += [[k, v]]
            else:
                default_param_group += [[k, v]]

        # DEFAULT USAGE FOR YOLO TRAINING IS WITH NESTEROV
        nesterov = True if 'nesterov' not in optimizer_params.keys() else optimizer_params['nesterov']

        default_param_group_optimizer_format = {'named_params': default_param_group,
                                                'lr': lr,
                                                'nesterov': nesterov,
                                                'momentum': optimizer_params['momentum']}

        weight_decay_param_group_optimizer_format = {'named_params': weight_decay_param_group,
                                                     'weight_decay': optimizer_params['weight_decay']}

        biases_param_group_optimizer_format = {'named_params': biases_param_group}

        return [default_param_group_optimizer_format,
                weight_decay_param_group_optimizer_format,
                biases_param_group_optimizer_format]

    @staticmethod
    def concatenate_layer_output(x, layer_index: int, route_layers: list) -> tuple:
        """
        concatenate_layer_output
            :param x:               input for the layer
            :param layer_index:     the layer index to decide how to concatenate to
            :param route_layers:    the route layers list with previous data
            :return:                tuple of x, route_layers
        """
        # CONCATENATE THE OUTPUTS OF PREVIOUS LAYERS
        if layer_index in [6, 8, 16, 26]:
            route_layers.append(x)

        if layer_index == 19:
            x = route_layers[2]
        if layer_index == 29:
            x = route_layers[3]
        if layer_index == 21:
            x = torch.cat((x, route_layers[1]), 1)
        if layer_index == 31:
            x = torch.cat((x, route_layers[0]), 1)

        return x, route_layers

    def get_yolo_layers_indices(self):
        return [i for i, module in enumerate(self.module_list) if isinstance(module, YOLOLayer)]

    @staticmethod
    def add_yolo_layer_to_modules_list(modules_list: nn.ModuleList, image_size: int, yolo_v3_anchors: list,
                                       anchors_mask: list, num_classes: int, onnx_stride: int,
                                       onnx_export_mode: bool = False) -> nn.ModuleList:
        """
        add_yolo_layer_to_modules_list - Adds a YoLo Head Layer to the nn.ModuleList
            :param modules_list:            The Modules List
            :param image_size:              The YoLo Model Image Size
            :param yolo_v3_anchors:         The Anchors (K-Means) List for the YoLo Layer Initialization
            :param anchors_mask:            the mask to get the relevant anchors
            :param num_classes:             The number of different classes in the data
            :param onnx_stride:             The stride of the layer for ONNX grid points calculation in YoLo Layer init
            :param onnx_export_mode:        Alter the model YoLo Layer for ONNX Export
            :return:                        The nn.ModuleList with the Added Yolo layer, and a Bias Initialization
        """
        mask = [yolo_v3_anchors[i] for i in anchors_mask]

        b = [-5.5, -5.0]
        bias = modules_list[-1][0].bias.view(len(mask), -1)  # PRE-YOLO-LAYER to 3x(NUM_CLASSES + 5)
        with torch.no_grad():
            bias[:, 4] += b[0] - bias[:, 4].mean()  # OBJECTNESS
            bias[:, 5:] += b[1] - bias[:, 5:].mean()  # CLASSIFICATION
            modules_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))

        modules_list.append(YOLOLayer(anchors_mask=anchors_mask, classes_num=num_classes, anchors=yolo_v3_anchors,
                                      image_size=image_size, onnx_stride=onnx_stride,
                                      onnx_export_mode=onnx_export_mode))

        return modules_list

    @staticmethod
    def named_sequential_module(module_name, module) -> nn.Sequential:
        """
        create_named_nn_sequential_module
            :param module_name:
            :param module:
            :return: nn.Sequential() with the added relevant names
        """
        named_sequential_module = nn.Sequential()
        named_sequential_module.add_module(module_name, module)
        return named_sequential_module

    def create_modules_list(self, num_classes: int):
        """
        create_modules_list
            :param num_classes:
            :return:
        """
        # DARKNET BACKBONE ARCHITECTURE
        darknet_53 = Darknet53(backbone_mode=True)
        yolo_modules_list = darknet_53.get_modules_list()

        # YOLO V3 ARCHITECTURE
        yolo_modules_list.append(DarkResidualBlock(in_channels=1024, shortcut=False))  # 11
        yolo_modules_list.append(create_conv_module(in_channels=1024, out_channels=512, kernel_size=1, stride=1))  # 12
        yolo_modules_list.append(SPPLayer())  # 13
        yolo_modules_list.append(create_conv_module(in_channels=2048, out_channels=512, kernel_size=1, stride=1))  # 14
        yolo_modules_list.append(create_conv_module(in_channels=512, out_channels=1024, kernel_size=3, stride=1))  # 15
        yolo_modules_list.append(create_conv_module(in_channels=1024, out_channels=512, kernel_size=1, stride=1))  # 16
        yolo_modules_list.append(create_conv_module(in_channels=512, out_channels=1024, kernel_size=3, stride=1))  # 17

        yolo_modules_list.append(self.named_sequential_module('Conv2d',
                                                              nn.Conv2d(in_channels=1024,
                                                                        out_channels=self.pre_yolo_layer_size,
                                                                        kernel_size=1, stride=1)))  # 18

        yolo_modules_list = self.add_yolo_layer_to_modules_list(modules_list=yolo_modules_list,  # 19
                                                                image_size=self.image_size,
                                                                yolo_v3_anchors=self.yolo_v3_anchors,
                                                                anchors_mask=[6, 7, 8], num_classes=num_classes,
                                                                onnx_stride=32, onnx_export_mode=self.onnx_export_mode)

        yolo_modules_list.append(create_conv_module(in_channels=512, out_channels=256, kernel_size=1, stride=1))  # 20
        yolo_modules_list.append(Upsample(scale_factor=2, mode='nearest'))  # 21
        yolo_modules_list.append(create_conv_module(in_channels=768, out_channels=256, kernel_size=1, stride=1))  # 22
        yolo_modules_list.append(create_conv_module(in_channels=256, out_channels=512, kernel_size=3, stride=1))  # 23

        yolo_modules_list.append(create_conv_module(in_channels=512, out_channels=256, kernel_size=1, stride=1))  # 24
        yolo_modules_list.append(create_conv_module(in_channels=256, out_channels=512, kernel_size=3, stride=1))  # 25
        yolo_modules_list.append(create_conv_module(in_channels=512, out_channels=256, kernel_size=1, stride=1))  # 26
        yolo_modules_list.append(create_conv_module(in_channels=256, out_channels=512, kernel_size=3, stride=1))  # 27

        yolo_modules_list.append(self.named_sequential_module('Conv2d',
                                                              nn.Conv2d(in_channels=512,
                                                                        out_channels=self.pre_yolo_layer_size,
                                                                        kernel_size=1, stride=1)))  # 28

        yolo_modules_list = self.add_yolo_layer_to_modules_list(modules_list=yolo_modules_list,  # 29
                                                                image_size=self.image_size,
                                                                yolo_v3_anchors=self.yolo_v3_anchors,
                                                                anchors_mask=[3, 4, 5], num_classes=num_classes,
                                                                onnx_stride=16, onnx_export_mode=self.onnx_export_mode)

        yolo_modules_list.append(create_conv_module(in_channels=256, out_channels=128, kernel_size=1, stride=1))  # 30
        yolo_modules_list.append(Upsample(scale_factor=2, mode='nearest'))  # 31
        yolo_modules_list.append(create_conv_module(in_channels=384, out_channels=128, kernel_size=1, stride=1))  # 32
        yolo_modules_list.append(create_conv_module(in_channels=128, out_channels=256, kernel_size=3, stride=1))  # 33
        yolo_modules_list.append(create_conv_module(in_channels=256, out_channels=128, kernel_size=1, stride=1))  # 34
        yolo_modules_list.append(create_conv_module(in_channels=128, out_channels=256, kernel_size=3, stride=1))  # 35
        yolo_modules_list.append(create_conv_module(in_channels=256, out_channels=128, kernel_size=1, stride=1))  # 36
        yolo_modules_list.append(create_conv_module(in_channels=128, out_channels=256, kernel_size=3, stride=1))  # 37
        yolo_modules_list.append(self.named_sequential_module('Conv2d',
                                                              nn.Conv2d(in_channels=256,
                                                                        out_channels=self.pre_yolo_layer_size,
                                                                        kernel_size=1, stride=1)))  # 38

        yolo_modules_list = self.add_yolo_layer_to_modules_list(modules_list=yolo_modules_list,  # 39
                                                                image_size=self.image_size,
                                                                yolo_v3_anchors=self.yolo_v3_anchors,
                                                                anchors_mask=[0, 1, 2], num_classes=num_classes,
                                                                onnx_stride=8, onnx_export_mode=self.onnx_export_mode)

        return yolo_modules_list

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Method for preparing the Yolov3 and TinyYolov3 for conversion (ONNX, TRT, CoreML etc).
        :param input_size: used for calculating the grid points.
        """
        self.onnx_export_mode = True

        # ONNX EXPORT REQUIRES GRIDS TO BE CALCULATED IN init of YOLOLayer SO WE RE-RUN THE CALC METHOD
        for module in self.module_list:
            if isinstance(module, YOLOLayer):
                module.onnx_export_mode = True
                x_grid_points_num = int(input_size / module.onnx_stride)
                y_grid_points_num = int(input_size / module.onnx_stride)
                module.create_grids((input_size, input_size), (x_grid_points_num, y_grid_points_num))


class TinyYoloV3(YoloV3):
    """
    TinyYoloV3 - Inherits from YoLoV3 class and overloads the relevant methods and members
    """

    def __init__(self, num_classes: int = 80, image_size: int = 416,
                 arch_params: dict = None,
                 iou_t: float = 0.225, yolo_v3_anchors: list = None):

        if arch_params:
            yolo_v3_anchors = arch_params.yolo_v3_anchors if 'yolo_v3_anchors' in arch_params.to_dict() else yolo_v3_anchors
        if yolo_v3_anchors is None:
            # DEFAULT ANCHORS FOR TINY YOLO V3
            yolo_v3_anchors = [(10., 14.), (23., 27.), (37., 58.),
                               (81., 82.), (135., 169.), (344., 319.)]

        super(TinyYoloV3, self).__init__(num_classes=num_classes, image_size=image_size,
                                         arch_params=arch_params, iou_t=iou_t, yolo_v3_anchors=yolo_v3_anchors)

    @staticmethod
    def concatenate_layer_output(x, layer_index: int, route_layers: list) -> tuple:
        """
        concatenate_layer_output
            :param x:               input for the layer
            :param layer_index:     the layer index to decide how to concatenate to
            :param route_layers:    the route layers list with previous data
            :return:                tuple of x, route_layers
        """
        # CONCATENATE THE OUTPUTS OF PREVIOUS LAYERS
        if layer_index in [8, 14]:
            route_layers.append(x)
        if layer_index == 17:
            x = route_layers[1]
        if layer_index == 19:
            x = torch.cat((x, route_layers[0]), 1)

        return x, route_layers

    def create_modules_list(self, num_classes: int):
        """
        create_tiny_modules_list
            :param num_classes:     The Number of different Classes
            :return:                nn.ModuleList with the Tiny-Yolo-V3 Architecture
        """
        yolo_modules_list = nn.ModuleList()

        yolo_modules_list.append(create_conv_module(3, 16))  # 0
        yolo_modules_list.append(self.named_sequential_module('MaxPool2d', nn.MaxPool2d(stride=2, kernel_size=2)))  # 1
        yolo_modules_list.append(create_conv_module(16, 32))  # 2
        yolo_modules_list.append(self.named_sequential_module('MaxPool2d', nn.MaxPool2d(stride=2, kernel_size=2)))  # 3
        yolo_modules_list.append(create_conv_module(32, 64))  # 4
        yolo_modules_list.append(self.named_sequential_module('MaxPool2d', nn.MaxPool2d(stride=2, kernel_size=2)))  # 5
        yolo_modules_list.append(create_conv_module(64, 128))  # 6
        yolo_modules_list.append(self.named_sequential_module('MaxPool2d', nn.MaxPool2d(stride=2, kernel_size=2)))  # 7
        yolo_modules_list.append(create_conv_module(128, 256))  # 8
        yolo_modules_list.append(self.named_sequential_module('MaxPool2d', nn.MaxPool2d(stride=2, kernel_size=2)))  # 9
        yolo_modules_list.append(create_conv_module(256, 512))  # 10
        yolo_modules_list.append(self.named_sequential_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1))))  # 11
        yolo_modules_list.append(self.named_sequential_module('MaxPool2d', nn.MaxPool2d(stride=1, kernel_size=2)))  # 12
        yolo_modules_list.append(create_conv_module(512, 1024))  # 13
        yolo_modules_list.append(create_conv_module(1024, 256, kernel_size=1))  # 14
        yolo_modules_list.append(create_conv_module(256, 512))  # 15
        yolo_modules_list.append(self.named_sequential_module('Conv2d',
                                                              nn.Conv2d(in_channels=512,
                                                                        out_channels=self.pre_yolo_layer_size,
                                                                        kernel_size=1, stride=1)))  # 16

        yolo_modules_list = self.add_yolo_layer_to_modules_list(modules_list=yolo_modules_list,  # 17
                                                                image_size=self.image_size,
                                                                yolo_v3_anchors=self.yolo_v3_anchors,
                                                                anchors_mask=[3, 4, 5], num_classes=num_classes,
                                                                onnx_stride=32, onnx_export_mode=self.onnx_export_mode)

        yolo_modules_list.append(create_conv_module(256, 128, kernel_size=1))  # 18
        yolo_modules_list.append(Upsample(scale_factor=2, mode='nearest'))  # 19
        yolo_modules_list.append(create_conv_module(384, 256))  # 20
        yolo_modules_list.append(self.named_sequential_module('Conv2d',
                                                              nn.Conv2d(in_channels=256,
                                                                        out_channels=self.pre_yolo_layer_size,
                                                                        kernel_size=1, stride=1)))  # 21

        # THE [1, 2, 3] IN THE MASK IS NOT A BUG, BUT A FEATURE :)
        yolo_modules_list = self.add_yolo_layer_to_modules_list(modules_list=yolo_modules_list,  # 22
                                                                image_size=self.image_size,
                                                                yolo_v3_anchors=self.yolo_v3_anchors,
                                                                anchors_mask=[1, 2, 3], num_classes=num_classes,
                                                                onnx_stride=16, onnx_export_mode=self.onnx_export_mode)

        return yolo_modules_list
