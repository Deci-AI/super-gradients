# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from functools import lru_cache
from typing import Mapping, Any, Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.predict import ImagesPoseEstimationPrediction
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.models.arch_params_factory import get_arch_params

__all__ = ["DEKRPoseEstimationModel", "DEKRW32NODC"]

from super_gradients.training.pipelines.pipelines import PoseEstimationPipeline

from super_gradients.training.processing.processing import Processing

from super_gradients.training.utils import HpmStruct, DEKRPoseEstimationDecodeCallback
from super_gradients.training.utils.media.image import ImageSource

logger = get_logger(__name__)


class BasicBlock(nn.Module):
    """
    ResNet basic block
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet bottleneck block
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdaptBlock(nn.Module):
    """
    Residual block with deformable convolution
    """

    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, downsample=None, dilation=1, deformable_groups=1):
        super(AdaptBlock, self).__init__()
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        self.register_buffer("regular_matrix", regular_matrix.float())
        self.downsample = downsample
        self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)

        self.adapt_conv = torchvision.ops.DeformConv2d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False, groups=deformable_groups
        )

        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        N, _, H, W = x.shape
        transform_matrix = self.transform_matrix_conv(x)
        transform_matrix = transform_matrix.permute(0, 2, 3, 1).reshape((N * H * W, 2, 2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset - self.regular_matrix
        offset = offset.transpose(1, 2).reshape((N, H, W, 18)).permute(0, 3, 1, 2)

        translation = self.translation_conv(x)
        offset[:, 0::2, :, :] += translation[:, 0:1, :, :]
        offset[:, 1::2, :, :] += translation[:, 1:2, :, :]

        out = self.adapt_conv(x, offset)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck, "ADAPTIVE": AdaptBlock}


@register_model(Models.DEKR_CUSTOM)
class DEKRPoseEstimationModel(SgModule):
    """
    Implementation of HRNet model from DEKR paper (https://arxiv.org/abs/2104.02300).

    The model takes an image of (B,C,H,W) shape and outputs two tensors (heatmap, offset) as predictions:
      - heatmap (B, NumJoints+1,H * upsample_factor, W * upsample_factor)
      - offset (B, NumJoints*2, H * upsample_factor, W * upsample_factor)
    """

    def __init__(self, arch_params):
        super(DEKRPoseEstimationModel, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # build stage
        self.spec = arch_params.SPEC
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES
        num_channels_last = [256]
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = self._make_transition_layer(num_channels_last, num_channels)
            setattr(self, "transition{}".format(i + 1), transition_layer)

            stage, num_channels_last = self._make_stage(self.stages_spec, i, num_channels, True)
            setattr(self, "stage{}".format(i + 2), stage)

        # build head net
        inp_channels = int(sum(self.stages_spec.NUM_CHANNELS[-1]))
        config_heatmap = self.spec.HEAD_HEATMAP
        config_offset = self.spec.HEAD_OFFSET
        self.num_joints = arch_params.num_classes
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints + 1
        self.offset_prekpt = config_offset["NUM_CHANNELS_PERKPT"]

        offset_channels = self.num_joints * self.offset_prekpt
        self.transition_heatmap = self._make_transition_for_head(inp_channels, config_heatmap["NUM_CHANNELS"])
        self.transition_offset = self._make_transition_for_head(inp_channels, offset_channels)
        self.head_heatmap = self._make_heatmap_head(config_heatmap)
        self.offset_feature_layers, self.offset_final_layer = self._make_separete_regression_head(config_offset)
        self.heatmap_activation = nn.Sigmoid() if config_heatmap["HEATMAP_APPLY_SIGMOID"] else nn.Identity()
        self.init_weights()

    def _make_transition_for_head(self, inplanes: int, outplanes: int) -> nn.Module:
        transition_layer = [nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False), nn.BatchNorm2d(outplanes), nn.ReLU(True)]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, layer_config: Mapping[str, Any]) -> nn.ModuleList:
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config["BLOCK"]],
            layer_config["NUM_CHANNELS"],
            layer_config["NUM_CHANNELS"],
            layer_config["NUM_BLOCKS"],
            dilation=layer_config["DILATION_RATE"],
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config["NUM_CHANNELS"],
            out_channels=self.num_joints_with_center,
            kernel_size=self.spec.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0,
        )
        heatmap_head_layers.append(heatmap_conv)

        return nn.ModuleList(heatmap_head_layers)

    def _make_separete_regression_head(self, layer_config) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """
        Build offset regression head for each joint
        :param layer_config:
        :return:
        """
        offset_feature_layers = []
        offset_final_layer = []

        for _ in range(self.num_joints):
            feature_conv = self._make_layer(
                blocks_dict[layer_config["BLOCK"]],
                layer_config["NUM_CHANNELS_PERKPT"],
                layer_config["NUM_CHANNELS_PERKPT"],
                layer_config["NUM_BLOCKS"],
                dilation=layer_config["DILATION_RATE"],
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config["NUM_CHANNELS_PERKPT"],
                out_channels=2,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0,
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, stages_spec, stage_index, num_inchannels, multi_scale_output=True):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec["BLOCK"][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, "transition{}".format(i + 1))
            for j in range(self.stages_spec["NUM_BRANCHES"][i]):
                if transition[j]:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, "stage{}".format(i + 2))(x_list)

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x = torch.cat(
            [
                y_list[0],
                F.upsample(y_list[1], size=(x0_h, x0_w), mode="bilinear"),
                F.upsample(y_list[2], size=(x0_h, x0_w), mode="bilinear"),
                F.upsample(y_list[3], size=(x0_h, x0_w), mode="bilinear"),
            ],
            1,
        )

        heatmap = self.head_heatmap[1](self.head_heatmap[0](self.transition_heatmap(x)))

        final_offset = []
        offset_feature = self.transition_offset(x)

        for j in range(self.num_joints):
            final_offset.append(
                self.offset_final_layer[j](self.offset_feature_layers[j](offset_feature[:, j * self.offset_prekpt : (j + 1) * self.offset_prekpt]))
            )

        offset = torch.cat(final_offset, dim=1)
        return self.heatmap_activation(heatmap), offset

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, "transform_matrix_conv"):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, "bias"):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, "translation_conv"):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, "bias"):
                    nn.init.constant_(m.translation_conv.bias, 0)

    @staticmethod
    def get_post_prediction_callback(conf: float = 0.05):
        return DEKRPoseEstimationDecodeCallback(
            min_confidence=conf,
            keypoint_threshold=0.05,
            nms_threshold=0.05,
            apply_sigmoid=True,
            max_num_people=30,
            nms_num_threshold=8,
            output_stride=4,
        )

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        edge_links: Union[np.ndarray, List[Tuple[int, int]]],
        edge_colors: Union[np.ndarray, List[Tuple[int, int, int]]],
        keypoint_colors: Union[np.ndarray, List[Tuple[int, int, int]]],
        image_processor: Optional[Processing] = None,
        conf: Optional[float] = None,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param conf:            (Optional) Below the confidence threshold, prediction are discarded
        """
        self._edge_links = edge_links or self._edge_links
        self._edge_colors = edge_colors or self._edge_colors
        self._keypoint_colors = keypoint_colors or self._keypoint_colors
        self._image_processor = image_processor or self._image_processor
        self._default_nms_conf = conf or self._default_nms_conf

    @lru_cache(maxsize=1)
    def _get_pipeline(self, conf: Optional[float] = None, fuse_model: bool = True) -> PoseEstimationPipeline:
        """Instantiate the prediction pipeline of this model.

        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        if None in (self._edge_links, self._image_processor, self._default_nms_conf):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        conf = conf or self._default_nms_conf

        if len(self._keypoint_colors) != self.num_joints:
            raise RuntimeError(
                "The number of colors for the keypoints ({}) does not match the number of joints ({})".format(len(self._keypoint_colors), self.num_joints)
            )
        if len(self._edge_colors) != len(self._edge_links):
            raise RuntimeError(
                "The number of colors for the joints ({}) does not match the number of joint links ({})".format(len(self._edge_colors), len(self._edge_links))
            )

        pipeline = PoseEstimationPipeline(
            model=self,
            image_processor=self._image_processor,
            edge_links=self._edge_links,
            edge_colors=self._edge_colors,
            keypoint_colors=self._keypoint_colors,
            post_prediction_callback=self.get_post_prediction_callback(conf=conf),
            fuse_model=fuse_model,
        )
        return pipeline

    def predict(self, images: ImageSource, conf: Optional[float] = None, fuse_model: bool = True) -> ImagesPoseEstimationPrediction:
        """Predict an image or a list of images.

        :param images:  Images to predict.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(conf=conf, fuse_model=fuse_model)
        return pipeline(images)  # type: ignore

    def predict_webcam(self, conf: Optional[float] = None, fuse_model: bool = True):
        """Predict using webcam.

        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(conf=conf, fuse_model=fuse_model)
        pipeline.predict_webcam()

    def train(self, mode: bool = True):
        self._get_pipeline.cache_clear()
        torch.cuda.empty_cache()
        return super().train(mode)


@register_model(Models.DEKR_W32_NO_DC)
class DEKRW32NODC(DEKRPoseEstimationModel):
    """
    DEKR-W32 model for pose estimation without deformable convolutions.
    """

    def __init__(self, arch_params):
        POSE_DEKR_W32_NO_DC_ARCH_PARAMS = get_arch_params("pose_dekr_w32_no_dc_arch_params")

        merged_arch_params = HpmStruct(**copy.deepcopy(POSE_DEKR_W32_NO_DC_ARCH_PARAMS))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(merged_arch_params)


class DEKRWrapper(nn.Module):
    def __init__(self, model: DEKRPoseEstimationModel, apply_sigmoid=False):
        super().__init__()
        self.model = model
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs):
        heatmap, offsets = self.model(inputs)

        if self.apply_sigmoid:
            heatmap = torch.sigmoid(heatmap)

        return heatmap, offsets


class DEKRHorisontalFlipWrapper(nn.Module):
    def __init__(self, model: DEKRPoseEstimationModel, flip_indexes, apply_sigmoid=False):
        super().__init__()
        self.model = model
        # In DEKR the heatmap has one more channel for the center point of the pose, which is the last channel and it is not flipped
        self.flip_indexes_heatmap = torch.tensor(list(flip_indexes) + [len(flip_indexes)]).long()
        self.flip_indexes_offset = torch.tensor(flip_indexes).long()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs):

        input_flip = inputs.flip(3)
        input_flip[:, :, :, :-3] = input_flip[:, :, :, 3:]

        heatmap, offsets = self.model(inputs)
        heatmap_flip, offset_flip = self.model(input_flip)

        heatmap_deaugment = heatmap_flip[:, self.flip_indexes_heatmap, :, :]

        batch_size, num_offsets, rows, cols = offset_flip.size()

        offset_flip = offset_flip.reshape(offset_flip.size(0), offset_flip.size(1) // 2, 2, offset_flip.size(2), offset_flip.size(3))
        offset_flip = offset_flip[:, self.flip_indexes_offset, :, :, :]
        offset_flip[:, :, 0, :, :] *= -1

        offset_deaugment = offset_flip.reshape(batch_size, num_offsets, rows, cols)

        if self.apply_sigmoid:
            heatmap = torch.sigmoid(heatmap)
            heatmap_deaugment = torch.sigmoid(heatmap_deaugment)

        averaged_heatmap = (heatmap + heatmap_deaugment.flip(3)) * 0.5
        averaged_offsets = (offsets + offset_deaugment.flip(3)) * 0.5

        return averaged_heatmap, averaged_offsets
