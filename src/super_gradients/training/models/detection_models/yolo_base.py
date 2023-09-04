import collections
import math
import warnings
from typing import Union, Type, List, Tuple, Optional, Any
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.module_interfaces import ExportableObjectDetectionModel, AbstractObjectDetectionDecodingModule, HasPredict
from super_gradients.module_interfaces.exportable_detector import ModelHasNoPreprocessingParamsException
from super_gradients.modules import CrossModelSkipConnection, Conv
from super_gradients.training.models.classification_models.regnet import AnyNetX, Stage
from super_gradients.training.models.detection_models.csp_darknet53 import GroupedConvBlock, CSPDarknet53, get_yolo_type_params, SPP
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils import torch_version_is_greater_or_equal
from super_gradients.training.utils.detection_utils import (
    non_max_suppression,
    matrix_non_max_suppression,
    NMS_Type,
    DetectionPostPredictionCallback,
    Anchors,
    convert_cxcywh_bbox_to_xyxy,
)
from super_gradients.training.utils.utils import HpmStruct, check_img_size_divisibility, get_param, infer_model_dtype, infer_model_device
from super_gradients.training.utils.predict import ImagesDetectionPrediction
from super_gradients.training.pipelines.pipelines import DetectionPipeline
from super_gradients.training.processing.processing import Processing
from super_gradients.training.utils.media.image import ImageSource


COCO_DETECTION_80_CLASSES_BBOX_ANCHORS = Anchors(
    [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], strides=[8, 16, 32]
)  # output strides of all yolo outputs

ANCHORSLESS_DUMMY_ANCHORS = Anchors([[0, 0], [0, 0], [0, 0]], strides=[8, 16, 32])


DEFAULT_YOLO_ARCH_PARAMS = {
    "num_classes": 80,  # Number of classes to predict
    "depth_mult_factor": 1.0,  # depth multiplier for the entire model
    "width_mult_factor": 1.0,  # width multiplier for the entire model
    "channels_in": 3,  # Number of channels in the input image
    "skip_connections_list": [(12, [6]), (16, [4]), (19, [14]), (22, [10]), (24, [17, 20])],
    # A list defining skip connections. format is '[target: [source1, source2, ...]]'. Each item defines a skip
    # connection from all sources to the target according to the layer's index (count starts from the backbone)
    "backbone_connection_channels": [1024, 512, 256],  # width of backbone channels that are concatenated with the head
    # True if width_mult_factor is applied to the backbone (is the case with the default backbones)
    # which means that backbone_connection_channels should be used with a width_mult_factor
    # False if backbone_connection_channels should be used as is
    "scaled_backbone_width": True,
    "fuse_conv_and_bn": False,  # Fuse sequential Conv + B.N layers into a single one
    "add_nms": False,  # Add the NMS module to the computational graph
    "nms_conf": 0.25,  # When add_nms is True during NMS predictions with confidence lower than this will be discarded
    "nms_iou": 0.45,  # When add_nms is True IoU threshold for NMS algorithm
    # (with smaller value more boxed will be considered "the same" and removed)
    "yolo_type": "yoloX",  # Type of yolo to build: 'yoloX' is only supported currently
    "stem_type": None,  # 'focus' and '6x6' are supported, by default is defined by yolo_type and yolo_version
    "depthwise": False,  # use depthwise separable convolutions all over the model
    "xhead_inter_channels": None,  # (has an impact only if yolo_type is yoloX)
    # Channels in classification and regression branches of the detecting blocks;
    # if is None the first of input channels will be used by default
    "xhead_groups": None,  # (has an impact only if yolo_type is yoloX)
    # Num groups in convs in classification and regression branches of the detecting blocks;
    # if None default groups will be used according to conv type
    # (1 for Conv and depthwise for GroupedConvBlock)
}


class YoloXPostPredictionCallback(DetectionPostPredictionCallback):
    """Post-prediction callback to decode YoloX model's output and apply Non-Maximum Suppression (NMS) to get
    the final predictions.
    """

    def __init__(
        self,
        conf: float = 0.001,
        iou: float = 0.6,
        classes: List[int] = None,
        nms_type: NMS_Type = NMS_Type.ITERATIVE,
        max_predictions: int = 300,
        with_confidence: bool = True,
        class_agnostic_nms: bool = False,
        multi_label_per_box: bool = True,
    ):
        """
        :param conf: confidence threshold
        :param iou: IoU threshold                                       (used in NMS_Type.ITERATIVE)
        :param classes: (optional list) filter by class                 (used in NMS_Type.ITERATIVE)
        :param nms_type: the type of nms to use (iterative or matrix)
        :param max_predictions: maximum number of boxes to output       (used in NMS_Type.MATRIX)
        :param with_confidence: in NMS, whether to multiply objectness  (used in NMS_Type.ITERATIVE)
                                score with class score
        :param class_agnostic_nms: indicates how boxes of different classes will be treated during
                                   NMS step (used in NMS_Type.ITERATIVE and NMS_Type.MATRIX)
                                   True - NMS will be performed on all classes together.
                                   False - NMS will be performed on each class separately (default).
        :param multi_label_per_box: controls whether to decode multiple labels per box (used in NMS_Type.ITERATIVE)
                                    True - each anchor can produce multiple labels of different classes
                                           that pass confidence threshold check (default).
                                    False - each anchor can produce only one label of the class with the highest score.
        """
        super(YoloXPostPredictionCallback, self).__init__()
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.nms_type = nms_type
        self.max_pred = max_predictions
        self.with_confidence = with_confidence
        self.class_agnostic_nms = class_agnostic_nms
        self.multi_label_per_box = multi_label_per_box

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]], device: str = None):
        """Apply NMS to the raw output of the model and keep only top `max_predictions` results.

        :param x: Raw output of the model, with x[0] expected to be a list of Tensors of shape (cx, cy, w, h, confidence, cls0, cls1, ...)
        :return: List of Tensors of shape (x1, y1, x2, y2, conf, cls)
        """
        # Use the main output features in case of multiple outputs.
        if isinstance(x, (tuple, list)):
            x = x[0]

        if self.nms_type == NMS_Type.ITERATIVE:
            nms_result = non_max_suppression(
                x,
                conf_thres=self.conf,
                iou_thres=self.iou,
                with_confidence=self.with_confidence,
                multi_label_per_box=self.multi_label_per_box,
                class_agnostic_nms=self.class_agnostic_nms,
            )
        else:
            nms_result = matrix_non_max_suppression(x, conf_thres=self.conf, max_num_of_detections=self.max_pred, class_agnostic_nms=self.class_agnostic_nms)

        return self._filter_max_predictions(nms_result)

    def _filter_max_predictions(self, res: List) -> List:
        res[:] = [im[: self.max_pred] if (im is not None and im.shape[0] > self.max_pred) else im for im in res]
        return res


class YoloPostPredictionCallback(YoloXPostPredictionCallback):
    def __init__(
        self,
        conf: float = 0.001,
        iou: float = 0.6,
        classes: List[int] = None,
        nms_type: NMS_Type = NMS_Type.ITERATIVE,
        max_predictions: int = 300,
        with_confidence: bool = True,
    ):
        warnings.warn("YoloPostPredictionCallback is deprecated since SG 3.1.3, please use YoloXPostPredictionCallback instead", DeprecationWarning)
        super().__init__(
            conf=conf,
            iou=iou,
            classes=classes,
            nms_type=nms_type,
            max_predictions=max_predictions,
            with_confidence=with_confidence,
            class_agnostic_nms=False,
            multi_label_per_box=True,
        )


class Concat(nn.Module):
    """CONCATENATE A LIST OF TENSORS ALONG DIMENSION"""

    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, self.dimension)


class DetectX(nn.Module):
    def __init__(
        self,
        num_classes: int,
        stride: np.ndarray,
        activation_func_type: type,
        channels: list,
        depthwise=False,
        groups: int = None,
        inter_channels: Union[int, List] = None,
    ):
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

        if torch.is_tensor(stride):
            stride = stride.clone().detach()
        else:
            stride = torch.tensor(stride)

        self.register_buffer("stride", stride, persistent=False)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        ConvBlock = GroupedConvBlock if depthwise else Conv

        inter_channels = inter_channels or channels[0]
        inter_channels = inter_channels if isinstance(inter_channels, list) else [inter_channels] * self.detection_layers_num
        for i in range(self.detection_layers_num):
            self.stems.append(Conv(channels[i], inter_channels[i], 1, 1, activation_func_type))

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        ConvBlock(inter_channels[i], inter_channels[i], 3, 1, activation_func_type, groups=groups),
                        ConvBlock(inter_channels[i], inter_channels[i], 3, 1, activation_func_type, groups=groups),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        ConvBlock(inter_channels[i], inter_channels[i], 3, 1, activation_func_type, groups=groups),
                        ConvBlock(inter_channels[i], inter_channels[i], 3, 1, activation_func_type, groups=groups),
                    ]
                )
            )

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
                    self.grid[i] = self._make_grid(nx, ny, dtype=reg_output.dtype).to(output.device)

                xy = (output[..., :2] + self.grid[i].to(output.device)) * self.stride[i]
                wh = torch.exp(output[..., 2:4]) * self.stride[i]
                output = torch.cat([xy, wh, output[..., 4:].sigmoid()], dim=4)
                output = output.view(bs, -1, output.shape[-1])

            outputs.append(output)

        return outputs if self.training else (torch.cat(outputs, 1), outputs_logits)

    @staticmethod
    def _make_grid(nx: int, ny: int, dtype: torch.dtype):
        if torch_version_is_greater_or_equal(1, 10):
            # https://github.com/pytorch/pytorch/issues/50276
            yv, xv = torch.meshgrid([torch.arange(ny, dtype=dtype), torch.arange(nx, dtype=dtype)], indexing="ij")
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, dtype=dtype), torch.arange(nx, dtype=dtype)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).to(dtype)


class AbstractYoloBackbone:
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
            if layer_idx in self._layer_idx_to_extract:
                extracted_intermediate_layers.append(x)
            else:
                extracted_intermediate_layers.append(None)

        return extracted_intermediate_layers


class YoloDarknetBackbone(AbstractYoloBackbone, CSPDarknet53):
    """Implements the CSP_Darknet53 module and inherit the forward pass to extract layers indicated in arch_params"""

    def __init__(self, arch_params):
        arch_params.backbone_mode = True
        CSPDarknet53.__init__(self, arch_params)
        AbstractYoloBackbone.__init__(self, arch_params)

    def forward(self, x):
        return AbstractYoloBackbone.forward(self, x)


class YoloRegnetBackbone(AbstractYoloBackbone, AnyNetX):
    """Implements the Regnet module and inherits the forward pass to extract layers indicated in arch_params"""

    def __init__(self, arch_params):
        backbone_params = {**arch_params.backbone_params, "backbone_mode": True, "num_classes": None}
        backbone_params.pop("spp_kernels", None)
        AnyNetX.__init__(self, **backbone_params)

        # LAST ANYNETX STAGE -> STAGE + SPP IF SPP_KERNELS IS GIVEN
        spp_kernels = get_param(arch_params.backbone_params, "spp_kernels", None)
        if spp_kernels:
            activation_type = nn.SiLU if arch_params.yolo_type == "yoloX" else nn.Hardswish
            self.net.stage_3 = self.add_spp_to_stage(self.net.stage_3, spp_kernels, activation_type=activation_type)
            self.initialize_weight()

        # CREATE A LIST CONTAINING THE LAYERS TO EXTRACT FROM THE BACKBONE AND ADD THE FINAL LAYER
        self._modules_list = nn.ModuleList()
        for layer in self.net:
            self._modules_list.append(layer)

        AbstractYoloBackbone.__init__(self, arch_params)

        # WE KEEP A LIST OF THE OUTPUTS WIDTHS (NUM FEATURES) TO BE CONNECTED TO THE HEAD
        self.backbone_connection_channels = arch_params.backbone_params["ls_block_width"][1:][::-1]

    @staticmethod
    def add_spp_to_stage(anynetx_stage: Stage, spp_kernels: Tuple[int], activation_type):
        """
        Add SPP in the end of an AnyNetX Stage
        """
        # Last block in a Stage -> conv_block_3 -> Conv2d -> out_channels
        out_channels = anynetx_stage.blocks[-1].conv_block_3[0].out_channels
        anynetx_stage.blocks.add_module("spp_block", SPP(out_channels, out_channels, spp_kernels, activation_type=activation_type))
        return anynetx_stage

    def forward(self, x):
        return AbstractYoloBackbone.forward(self, x)


class YoloHead(nn.Module):
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

        _, block, activation_type, width_mult, depth_mult = get_yolo_type_params(
            arch_params.yolo_type, arch_params.width_mult_factor, arch_params.depth_mult_factor
        )

        backbone_connector = [width_mult(c) if arch_params.scaled_backbone_width else c for c in arch_params.backbone_connection_channels]

        DownConv = GroupedConvBlock if depthwise else Conv

        self._modules_list = nn.ModuleList()
        self._modules_list.append(Conv(backbone_connector[0], width_mult(512), 1, 1, activation_type))  # 10
        self._modules_list.append(nn.Upsample(None, 2, "nearest"))  # 11
        self._modules_list.append(Concat(1))  # 12
        self._modules_list.append(block(backbone_connector[1] + width_mult(512), width_mult(512), depth_mult(3), activation_type, False, depthwise))  # 13

        self._modules_list.append(Conv(width_mult(512), width_mult(256), 1, 1, activation_type))  # 14
        self._modules_list.append(nn.Upsample(None, 2, "nearest"))  # 15
        self._modules_list.append(Concat(1))  # 16
        self._modules_list.append(block(backbone_connector[2] + width_mult(256), width_mult(256), depth_mult(3), activation_type, False, depthwise))  # 17

        self._modules_list.append(DownConv(width_mult(256), width_mult(256), 3, 2, activation_type))  # 18
        self._modules_list.append(Concat(1))  # 19
        self._modules_list.append(block(2 * width_mult(256), width_mult(512), depth_mult(3), activation_type, False, depthwise))  # 20

        self._modules_list.append(DownConv(width_mult(512), width_mult(512), 3, 2, activation_type))  # 21
        self._modules_list.append(Concat(1))  # 22
        self._modules_list.append(block(2 * width_mult(512), width_mult(1024), depth_mult(3), activation_type, False, depthwise))  # 23

        detect_input_channels = [width_mult(v) for v in (256, 512, 1024)]
        strides = anchors.stride
        self._modules_list.append(
            DetectX(
                num_classes,
                strides,
                activation_type,
                channels=detect_input_channels,
                depthwise=depthwise,
                groups=xhead_groups,
                inter_channels=xhead_inter_channels,
            )
        )  # 24

        self._shortcuts = nn.ModuleList([CrossModelSkipConnection() for _ in range(len(self._skip_connections_dict.keys()) - 1)])

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
        i = 0
        for layer_idx, layer_module in enumerate(self._modules_list[:-1], start=num_layers_in_backbone):
            # IF THE LAYER APPEARS IN THE KEYS IT INSERT THE PRECIOUS OUTPUT AND THE INDICATED SKIP CONNECTIONS

            if layer_idx in self._skip_connections_dict.keys():
                out = layer_module([out, self._shortcuts[i](intermediate_output[self._skip_connections_dict[layer_idx][0]])])
                i += 1
            else:
                out = layer_module(out)

            # IF INDICATED APPEND THE OUTPUT TO inter_layer_idx_to_extract O.W. APPEND None
            if layer_idx in self._layer_idx_to_extract:
                intermediate_output.append(out)
            else:
                intermediate_output.append(None)

        # INSERT THE REMAINING LAYERS INTO THE Detect LAYER
        last_idx = len(self._modules_list) + num_layers_in_backbone - 1

        return self._modules_list[-1](
            [
                intermediate_output[self._skip_connections_dict[last_idx][0]],
                intermediate_output[self._skip_connections_dict[last_idx][1]],
                out,
            ]
        )


class YoloBase(SgModule, ExportableObjectDetectionModel, HasPredict):
    def __init__(self, backbone: Type[nn.Module], arch_params: HpmStruct, initialize_module: bool = True):
        super().__init__()
        # DEFAULT PARAMETERS TO BE OVERWRITTEN BY DUPLICATES THAT APPEAR IN arch_params
        self.arch_params = HpmStruct(**DEFAULT_YOLO_ARCH_PARAMS)
        # FIXME: REMOVE anchors ATTRIBUTE, WHICH HAS NO MEANING OTHER THAN COMPATIBILITY.
        self.arch_params.anchors = COCO_DETECTION_80_CLASSES_BBOX_ANCHORS
        self.arch_params.override(**arch_params.to_dict())
        self.arch_params.skip_connections_dict = {k: v for k, v in self.arch_params.skip_connections_list}
        self.in_channels = 3

        self.num_classes = self.arch_params.num_classes
        # THE MODEL'S MODULES
        self._backbone = backbone(arch_params=self.arch_params)
        if hasattr(self._backbone, "backbone_connection_channels"):
            self.arch_params.scaled_backbone_width = False
            self.arch_params.backbone_connection_channels = self._backbone.backbone_connection_channels

        self._nms = nn.Identity()

        # A FLAG TO DEFINE augment_forward IN INFERENCE
        self.augmented_inference = False

        if initialize_module:
            self._head = YoloHead(self.arch_params)
            self._initialize_module()

        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None
        self._default_nms_iou: Optional[float] = None
        self._default_nms_conf: Optional[float] = None
        self.register_buffer("strides", torch.tensor(self.arch_params.anchors.stride), persistent=False)

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> DetectionPostPredictionCallback:
        return YoloXPostPredictionCallback(conf=conf, iou=iou)

    def get_input_channels(self) -> int:
        return self.in_channels

    def get_processing_params(self) -> Optional[Processing]:
        return self._image_processor

    def get_preprocessing_callback(self, **kwargs):
        processing = self.get_processing_params()
        if processing is None:
            raise ModelHasNoPreprocessingParamsException()
        preprocessing_module = processing.get_equivalent_photometric_module()
        return preprocessing_module

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        class_names: Optional[List[str]] = None,
        image_processor: Optional[Processing] = None,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param class_names:     (Optional) Names of the dataset the model was trained on.
        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param iou:             (Optional) IoU threshold for the nms algorithm
        :param conf:            (Optional) Below the confidence threshold, prediction are discarded
        """
        self._class_names = class_names or self._class_names
        self._image_processor = image_processor or self._image_processor
        self._default_nms_iou = iou or self._default_nms_iou
        self._default_nms_conf = conf or self._default_nms_conf

    @lru_cache(maxsize=1)
    def _get_pipeline(self, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True) -> DetectionPipeline:
        """Instantiate the prediction pipeline of this model.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        if None in (self._class_names, self._image_processor, self._default_nms_iou, self._default_nms_conf):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        iou = iou or self._default_nms_iou
        conf = conf or self._default_nms_conf

        pipeline = DetectionPipeline(
            model=self,
            image_processor=self._image_processor,
            post_prediction_callback=self.get_post_prediction_callback(iou=iou, conf=conf),
            class_names=self._class_names,
            fuse_model=fuse_model,
        )
        return pipeline

    def predict(
        self,
        images: ImageSource,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
    ) -> ImagesDetectionPrediction:
        """Predict an image or a list of images.

        :param images:      Images to predict.
        :param iou:         (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:        (Optional) Below the confidence threshold, prediction are discarded.
                            If None, the default value associated to the training is used.
        :param batch_size:  Maximum number of images to process at the same time.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(iou=iou, conf=conf, fuse_model=fuse_model)
        return pipeline(images, batch_size=batch_size)  # type: ignore

    def predict_webcam(self, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True):
        """Predict using webcam.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(iou=iou, conf=conf, fuse_model=fuse_model)
        pipeline.predict_webcam()

    def train(self, mode: bool = True):
        self._get_pipeline.cache_clear()
        torch.cuda.empty_cache()
        return super().train(mode)

    def forward(self, x):
        out = self._backbone(x)
        out = self._head(out)
        # THIS HAS NO EFFECT IF add_nms() WAS NOT DONE
        out = self._nms(out)
        return out

    def load_state_dict(self, state_dict, strict=True):
        try:
            keys_dropped_in_sg_320 = {
                "stride",
                "_head.anchors._stride",
                "_head.anchors._anchors",
                "_head.anchors._anchor_grid",
                "_head._modules_list.14.stride",
            }
            state_dict = collections.OrderedDict([(k, v) for k, v in state_dict.items() if k not in keys_dropped_in_sg_320])

            super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            raise RuntimeError(
                f"Got exception {e}, if a mismatch between expected and given state_dict keys exist, "
                f"checkpoint may have been saved after fusing conv and bn. use fuse_conv_bn before loading."
            )

    def _initialize_module(self):
        self._check_strides()
        self._initialize_biases()
        self._initialize_weights()
        if self.arch_params.add_nms:
            self._nms = self.get_post_prediction_callback(conf=self.arch_params.nms_conf, iou=self.arch_params.nms_iou)

    def _check_strides(self):
        m = self._head._modules_list[-1]  # DetectX()
        # Do inference in train mode on a dummy image to get output stride of each head output layer
        s = 128  # twice the minimum acceptable image size
        device = infer_model_device(m)
        dtype = infer_model_dtype(m)

        dummy_input = torch.zeros((1, self.arch_params.channels_in, s, s), device=device, dtype=dtype)
        dummy_input = dummy_input.to(next(self._backbone.parameters()).device)

        stride = torch.tensor([s / x.shape[-2] for x in self.forward(dummy_input)])
        stride = stride.to(m.stride.device)
        if not torch.equal(m.stride, stride):
            raise RuntimeError("Provided anchor strides do not match the model strides")

    def _initialize_biases(self):
        """initialize biases into DetectX()"""
        detect_module = self._head._modules_list[-1]  # DetectX() module
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

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        A method for preparing the Yolo model for conversion to other frameworks (ONNX, CoreML etc)
        :param input_size: expected input size
        :return:
        """
        assert not self.training, "model has to be in eval mode to be converted"

        # Verify dummy_input from converter is of multiple of the grid size
        max_stride = int(max(self.strides))

        # Validate the image size
        image_dims = input_size[-2:]  # assume torch uses channels first layout
        for dim in image_dims:
            res_flag, suggestion = check_img_size_divisibility(dim, max_stride)
            if not res_flag:
                raise ValueError(
                    f"Invalid input size: {input_size}. The input size must be multiple of max stride: "
                    f"{max_stride}. The closest suggestions are: {suggestion[0]}x{suggestion[0]} or "
                    f"{suggestion[1]}x{suggestion[1]}"
                )

    def get_include_attributes(self) -> list:
        return ["grid", "anchors", "anchors_grid"]

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self._head = new_head
        else:
            self.arch_params.num_classes = new_num_classes
            self.num_classes = new_num_classes
            old_detectx = self._head._modules_list[-1]
            _, block, activation_type, width_mult, depth_mult = get_yolo_type_params(
                self.arch_params.yolo_type, self.arch_params.width_mult_factor, self.arch_params.depth_mult_factor
            )

            new_last_layer = DetectX(
                num_classes=new_num_classes,
                stride=self.strides,
                activation_func_type=activation_type,
                channels=[width_mult(v) for v in (256, 512, 1024)],
                depthwise=isinstance(old_detectx.cls_convs[0][0], GroupedConvBlock),
                groups=self.arch_params.xhead_groups,
                inter_channels=self.arch_params.xhead_inter_channels,
            )
            new_last_layer = new_last_layer.to(next(self.parameters()).device)
            self._head._modules_list[-1] = new_last_layer
            self._check_strides()
            self._initialize_biases()
            self._initialize_weights()

    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractObjectDetectionDecodingModule:
        return YoloXDecodingModule(num_pre_nms_predictions=num_pre_nms_predictions, **kwargs)


class YoloXDecodingModule(AbstractObjectDetectionDecodingModule):
    __constants__ = ["num_pre_nms_predictions", "with_confidence"]

    def __init__(self, num_pre_nms_predictions: int, with_confidence: bool = True):
        super().__init__()
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.with_confidence = with_confidence

    def forward(self, predictions):
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        cxcywh = predictions[:, :, :4]
        conf = predictions[:, :, 4:5]
        pred_scores = predictions[:, :, 5:]
        pred_bboxes = convert_cxcywh_bbox_to_xyxy(cxcywh)

        if self.with_confidence:
            pred_scores = pred_scores * conf

        pred_cls_conf, _ = torch.max(pred_scores, dim=2)
        nms_top_k = self.num_pre_nms_predictions
        topk_candidates = torch.topk(pred_cls_conf, dim=1, k=nms_top_k, largest=True, sorted=True)

        offsets = nms_top_k * torch.arange(pred_cls_conf.size(0), device=pred_cls_conf.device)
        flat_indices = topk_candidates.indices + offsets.reshape(pred_cls_conf.size(0), 1)
        flat_indices = torch.flatten(flat_indices)

        output_pred_bboxes = pred_bboxes.reshape(-1, pred_bboxes.size(2))[flat_indices, :].reshape(pred_bboxes.size(0), nms_top_k, pred_bboxes.size(2))
        output_pred_scores = pred_scores.reshape(-1, pred_scores.size(2))[flat_indices, :].reshape(pred_scores.size(0), nms_top_k, pred_scores.size(2))

        return output_pred_bboxes, output_pred_scores

    def get_num_pre_nms_predictions(self) -> int:
        return self.num_pre_nms_predictions

    @torch.jit.ignore
    def infer_total_number_of_predictions(self, predictions: Any) -> int:
        """

        :param inputs:
        :return:
        """
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        return predictions.size(1)
