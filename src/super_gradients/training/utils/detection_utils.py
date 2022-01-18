import math
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Union, Tuple

import cv2
from deprecated import deprecated
from scipy.cluster.vq import kmeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import numpy as np
from torch import nn
from torch.nn import functional as F
from super_gradients.common.abstractions.abstract_logger import get_logger
from omegaconf import ListConfig


def base_detection_collate_fn(batch):
    """
    Batch Processing helper function for detection training/testing.
    stacks the lists of images and targets into tensors and adds the image index to each target (so the targets could
    later be associated to the correct images)
         :param batch:   Input batch from the Dataset __get_item__ method
         :return:        batch with the transformed values
     """

    images_batch, labels_batch = list(zip(*batch))
    for i, labels in enumerate(labels_batch):
        # ADD TARGET IMAGE INDEX
        labels[:, 0] = i

    return torch.stack(images_batch, 0), torch.cat(labels_batch, 0)


def convert_xyxy_bbox_to_xywh(input_bbox):
    """
    convert_xyxy_bbox_to_xywh - Converts bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
        :param input_bbox:  input bbox
        :return:            Converted bbox
    """
    converted_bbox = torch.zeros_like(input_bbox) if isinstance(input_bbox, torch.Tensor) else np.zeros_like(input_bbox)
    converted_bbox[:, 0] = (input_bbox[:, 0] + input_bbox[:, 2]) / 2
    converted_bbox[:, 1] = (input_bbox[:, 1] + input_bbox[:, 3]) / 2
    converted_bbox[:, 2] = input_bbox[:, 2] - input_bbox[:, 0]
    converted_bbox[:, 3] = input_bbox[:, 3] - input_bbox[:, 1]
    return converted_bbox


def convert_xywh_bbox_to_xyxy(input_bbox: torch.Tensor):
    """
    Converts bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
        :param input_bbox:  input bbox either 2-dimensional (for all boxes of a single image) or 3-dimensional (for
                            boxes of a batch of images)
        :return:            Converted bbox in same dimensions as the original
    """
    need_squeeze = False
    # the input is always processed as a batch. in case it not a batch, it is unsqueezed, process and than squeeze back.
    if input_bbox.dim() < 3:
        need_squeeze = True
        input_bbox = input_bbox.unsqueeze(0)

    converted_bbox = torch.zeros_like(input_bbox) if isinstance(input_bbox, torch.Tensor) else np.zeros_like(input_bbox)
    converted_bbox[:, :, 0] = input_bbox[:, :, 0] - input_bbox[:, :, 2] / 2
    converted_bbox[:, :, 1] = input_bbox[:, :, 1] - input_bbox[:, :, 3] / 2
    converted_bbox[:, :, 2] = input_bbox[:, :, 0] + input_bbox[:, :, 2] / 2
    converted_bbox[:, :, 3] = input_bbox[:, :, 1] + input_bbox[:, :, 3] / 2

    # squeeze back if needed
    if need_squeeze:
        converted_bbox = converted_bbox[0]

    return converted_bbox


def calculate_wh_iou(box1, box2) -> float:
    """
    calculate_wh_iou - Gets the Intersection over Union of the w,h values of the bboxes
        :param box1:
        :param box2:
        :return: IOU
    """
    # RETURNS THE IOU OF WH1 TO WH2. WH1 IS 2, WH2 IS NX2
    box2 = box2.t()

    # W, H = BOX1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # INTERSECTION AREA
    intersection_area = torch.min(w1, w2) * torch.min(h1, h2)

    # UNION AREA
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - intersection_area

    return intersection_area / union_area


def _iou(CIoU: bool, DIoU: bool, GIoU: bool, b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2, eps):
    """
    Internal function for the use of calculate_bbox_iou_matrix and calculate_bbox_iou_elementwise functions
    DO NOT CALL THIS FUNCTIONS DIRECTLY - use one of the functions mentioned above
    """
    # Intersection area
    intersection_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                        (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = w1 * h1 + w2 * h2 - intersection_area + eps
    iou = intersection_area / union_area  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        if GIoU:
            c_area = cw * ch + eps  # convex area
            iou -= (c_area - union_area) / c_area  # GIoU
        # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
        if DIoU or CIoU:
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + eps
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                iou -= rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                iou -= (rho2 / c2 + v * alpha)  # CIoU
    return iou


def calculate_bbox_iou_matrix(box1, box2, x1y1x2y2=True, GIoU: bool = False, DIoU=False, CIoU=False, eps=1e-9):
    """
    calculate iou matrix containing the iou of every couple iuo(i,j) where i is in box1 and j is in box2
        :param box1: a 2D tensor of boxes (shape N x 4)
        :param box2: a 2D tensor of boxes (shape M x 4)
        :param x1y1x2y2: boxes format is x1y1x2y2 (True) or xywh where xy is the center (False)
        :return: a 2D iou matrix (shape NxM)
    """
    if box1.dim() > 1:
        box1 = box1.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:  # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    b1_x1, b1_y1, b1_x2, b1_y2 = b1_x1.unsqueeze(1), b1_y1.unsqueeze(1), b1_x2.unsqueeze(1), b1_y2.unsqueeze(1)

    return _iou(CIoU, DIoU, GIoU, b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2, eps)


def calculate_bbox_iou_elementwise(box1, box2, x1y1x2y2=True, GIoU: bool = False, DIoU=False, CIoU=False, eps=1e-9):
    """
    calculate elementwise iou of two bbox tensors
        :param box1: a 2D tensor of boxes (shape N x 4)
        :param box2: a 2D tensor of boxes (shape N x 4)
        :param x1y1x2y2: boxes format is x1y1x2y2 (True) or xywh where xy is the center (False)
        :return: a 1D iou tensor (shape N)
    """
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    return _iou(CIoU, DIoU, GIoU, b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2, eps)


def calc_bbox_iou_matrix(pred: torch.Tensor):
    """
    calculate iou for every pair of boxes in the boxes vector
    :param pred: a 3-dimensional tensor containing all boxes for a batch of images [N, num_boxes, 4], where
                 each box format is [x1,y1,x2,y2]
    :return: a 3-dimensional matrix where M_i_j_k is the iou of box j and box k of the i'th image in the batch
    """
    box = pred[:, :, :4]  #
    b1_x1, b1_y1 = box[:, :, 0].unsqueeze(1), box[:, :, 1].unsqueeze(1)
    b1_x2, b1_y2 = box[:, :, 2].unsqueeze(1), box[:, :, 3].unsqueeze(1)

    b2_x1 = b1_x1.transpose(2, 1)
    b2_x2 = b1_x2.transpose(2, 1)
    b2_y1 = b1_y1.transpose(2, 1)
    b2_y2 = b1_y2.transpose(2, 1)
    intersection_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                        (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - intersection_area
    ious = intersection_area / union_area
    return ious


def build_detection_targets(detection_net: nn.Module, targets: torch.Tensor):
    """
    build_detection_targets - Builds the outputs of the Detection NN
                              This function filters all of the targets that don't have a sufficient iou coverage
                              of the Model's pre-trained k-means anchors
                              The iou_threshold is a parameter of the NN Model
        :param detection_net:   The nn.Module of the Detection Algorithm
        :param targets:         targets (labels)
        :return:
    """
    # TARGETS = [image, class, x, y, w, h]
    targets_num = len(targets)
    target_classes, target_bbox, indices, anchor_vector_list = [], [], [], []
    reject, use_all_anchors = True, True

    for i in detection_net.yolo_layers_indices:
        yolo_layer_module = list(detection_net.module_list)[i]

        # GET NUMBER OF GRID POINTS AND ANCHOR VEC FOR THIS YOLO LAYER
        grid_points_num, anchor_vec = yolo_layer_module.grid_size, yolo_layer_module.anchor_vec

        # IOU OF TARGETS-ANCHORS
        iou_targets, anchors = targets, []
        gwh = iou_targets[:, 4:6] * grid_points_num
        if targets_num:
            iou = torch.stack([calculate_wh_iou(x, gwh) for x in anchor_vec], 0)

            if use_all_anchors:
                anchors_num = len(anchor_vec)
                anchors = torch.arange(anchors_num).view((-1, 1)).repeat([1, targets_num]).view(-1)
                iou_targets = targets.repeat([anchors_num, 1])
                gwh = gwh.repeat([anchors_num, 1])
            else:
                # USE ONLY THE BEST ANCHOR
                iou, anchors = iou.max(0)  # best iou and anchor

            # REJECT ANCHORS BELOW IOU_THRES (OPTIONAL, INCREASES P, LOWERS R)
            if reject:
                # IOU THRESHOLD HYPERPARAMETER
                j = iou.view(-1) > detection_net.iou_t
                iou_targets, anchors, gwh = iou_targets[j], anchors[j], gwh[j]

        # INDICES
        target_image, target_class = iou_targets[:, :2].long().t()
        x_y_grid = iou_targets[:, 2:4] * grid_points_num
        x_grid_idx, y_grid_idx = x_y_grid.long().t()
        indices.append((target_image, anchors, y_grid_idx, x_grid_idx))

        # GIoU
        x_y_grid -= x_y_grid.floor()
        target_bbox.append(torch.cat((x_y_grid, gwh), 1))
        anchor_vector_list.append(anchor_vec[anchors])

        # Class
        target_classes.append(target_class)
        if target_class.shape[0]:
            if not target_class.max() < detection_net.num_classes:
                raise ValueError('Labeled Class is out of bounds of the classes list')

    return target_classes, target_bbox, indices, anchor_vector_list


def yolo_v3_non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5, device='cpu'):  # noqa: C901
    """
    non_max_suppression - Removes detections with lower object confidence score than 'conf_thres'
                          Non-Maximum Suppression to further filter detections.
        :param prediction:      the raw prediction as produced by the yolo_v3 network
        :param conf_thres:      confidence threshold - only prediction with confidence score higher than the threshold
                                will be considered
        :param nms_thres:       IoU threshold for the nms algorithm
        :param device:          the device to move all output tensors into
        :return:  (x1, y1, x2, y2, object_conf, class_conf, class)
    """
    # MINIMUM AND MAXIMIUM BOX WIDTH AND HEIGHT IN PIXELS
    min_wh = 2
    max_wh = 10000

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # MULTIPLY CONF BY CLASS CONF TO GET COMBINED CONFIDENCE
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # IGNORE ANYTHING UNDER conf_thres
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & (pred[:, 2:4] < max_wh).all(1) & \
            torch.isfinite(pred).all(1)
        pred = pred[i]

        # NOTHING IS OVER THE THRESHOLD
        if len(pred) == 0:
            continue

        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # BOX (CENTER X, CENTER Y, WIDTH, HEIGHT) TO (X1, Y1, X2, Y2)
        pred[:, :4] = convert_xywh_bbox_to_xyxy(pred[:, :4])

        # DETECTIONS ORDERED AS (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # SORT DETECTIONS BY DECREASING CONFIDENCE SCORES
        pred = pred[(-pred[:, 4]).argsort()]

        # 'OR', 'AND', 'MERGE', 'VISION', 'VISION_BATCHED'
        # MERGE is highest mAP, VISION is fastest
        method = 'MERGE' if conf_thres <= 0.01 else 'VISION'

        # BATCHED NMS
        if method == 'VISION_BATCHED':
            i = torchvision.ops.boxes.batched_nms(boxes=pred[:, :4],
                                                  scores=pred[:, 4],
                                                  idxs=pred[:, 6],
                                                  iou_threshold=nms_thres)
            output[image_i] = pred[i]
            continue

        # Non-maximum suppression
        det_max = []
        for detection_class in pred[:, -1].unique():
            dc = pred[pred[:, -1] == detection_class]
            n = len(dc)
            if n == 1:
                # NO NMS REQUIRED FOR A SINGLE CLASS
                det_max.append(dc)
                continue
            elif n > 500:
                dc = dc[:500]

            if method == 'VISION':
                i = torchvision.ops.boxes.nms(dc[:, :4], dc[:, 4], nms_thres)
                det_max.append(dc[i])

            elif method == 'OR':
                while dc.shape[0]:
                    det_max.append(dc[:1])
                    if len(dc) == 1:
                        break
                    iou = calculate_bbox_iou_elementwise(dc[0], dc[1:])
                    dc = dc[1:][iou < nms_thres]

            elif method == 'AND':
                while len(dc) > 1:
                    iou = calculate_bbox_iou_elementwise(dc[0], dc[1:])
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]

            elif method == 'MERGE':
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = calculate_bbox_iou_elementwise(dc[0], dc) > nms_thres
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif method == 'SOFT':
                sigma = 0.5
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = calculate_bbox_iou_elementwise(dc[0], dc[1:])
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)
                    dc = dc[dc[:, 4] > conf_thres]

        if len(det_max):
            det_max = torch.cat(det_max)
            output[image_i] = det_max[(-det_max[:, 4]).argsort()].to(device)

    return output


def change_bbox_bounds_for_image_size(boxes, img_shape):
    # CLIP BOUNDING XYXY BOUNDING BOXES TO IMAGE SHAPE (HEIGHT, WIDTH)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_shape[0])
    return boxes


def rescale_bboxes_for_image_size(current_image_shape, bbox, original_image_shape, ratio_pad=None):
    """
    rescale_bboxes_for_image_size - Changes the bboxes to fit the original image
        :param current_image_shape:
        :param bbox:
        :param original_image_shape:
        :param ratio_pad:
        :return:
    """
    if ratio_pad is None:
        gain = max(current_image_shape) / max(original_image_shape)
        # WH PADDING
        pad = (current_image_shape[1] - original_image_shape[1] * gain) / 2, \
              (current_image_shape[0] - original_image_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # X PADDING
    bbox[:, [0, 2]] -= pad[0]

    # Y PADDING
    bbox[:, [1, 3]] -= pad[1]
    bbox[:, :4] /= gain
    bbox = change_bbox_bounds_for_image_size(bbox, original_image_shape)
    return bbox


class DetectionPostPredictionCallback(ABC, nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x, device: str):
        """

        :param x:       the output of your model
        :param device:  the device to move all output tensors into
        :return:        a list with length batch_size, each item in the list is a detections
                        with shape: nx6 (x1, y1, x2, y2, confidence, class) where x and y are in range [0,1]
        """
        raise NotImplementedError


class YoloV3NonMaxSuppression(DetectionPostPredictionCallback):

    def __init__(self, conf: float = 0.001, nms_thres: float = 0.5, max_predictions=500) -> None:
        super().__init__()
        self.conf = conf
        self.max_predictions = max_predictions
        self.nms_thres = nms_thres

    def forward(self, x, device: str):
        return yolo_v3_non_max_suppression(x[0], device=device, conf_thres=self.conf, nms_thres=self.nms_thres)


class IouThreshold(tuple, Enum):
    MAP_05 = (0.5, 0.5)
    MAP_05_TO_095 = (0.5, 0.95)

    def is_range(self):
        return self[0] != self[1]


def scale_img(img, ratio=1.0, pad_to_original_img_size=False):
    """
    Scales the image by ratio (image dims is (batch_size, channels, height, width)
    Taken from Yolov5 Ultralitics repo"""
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        rescaled_size = (int(h * ratio), int(w * ratio))
        img = F.interpolate(img, size=rescaled_size, mode='bilinear', align_corners=False)
        # PAD THE IMAGE TO BE A MULTIPLIER OF grid_size. O.W. PAD IT TO THE ORIGINAL IMAGE SIZE
        if not pad_to_original_img_size:
            # THE MULTIPLIER WHICH THE DIMENSION MUST BE DIVISIBLE BY
            grid_size = 32
            # COMPUTE THE NEW SIZE OF THE IMAGE TO RETURN
            h, w = [math.ceil(x * ratio / grid_size) * grid_size for x in (h, w)]
        # PAD THE IMAGE TO FIT w, h (EITHER THE ORIGINAL SIZE OR THE NEW SIZE
        return F.pad(img, [0, w - rescaled_size[1], 0, h - rescaled_size[0]], value=0.447)  # value = imagenet mean


@deprecated(reason="use @torch.nn.utils.fuse_conv_bn_eval(conv, bn) instead")
def fuse_conv_and_bn(conv, bn):
    """Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    Taken from Yolov5 Ultralitics repo"""

    # init
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def check_anchor_order(m):
    """Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    Taken from Yolov5 Ultralitics repo"""
    anchor_area = m.anchor_grid.prod(-1).view(-1)
    delta_area = anchor_area[-1] - anchor_area[0]
    delta_stride = m.stride[-1] - m.stride[0]  # delta s
    # IF THE SIGN OF THE SUBTRACTION IS DIFFERENT => THE STRIDE IS NOT ALIGNED WITH ANCHORS => m.anchors ARE REVERSED
    if delta_area.sign() != delta_stride.sign():
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    Taken from Yolov5 Ultralitics repo
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None,
                        agnostic=False):  # noqa: C901
    """Performs Non-Maximum Suppression (NMS) on inference results
        :param prediction: raw model prediction
        :param conf_thres: below the confidence threshold - prediction are discarded
        :param iou_thres: IoU threshold for the nms algorithm
        :param merge: Merge boxes using weighted mean
        :param classes: (optional list) filter by class
        :param agnostic: Determines if is class agnostic. i.e. may display a box with 2 predictions
        :return:  (x1, y1, x2, y2, object_conf, class_conf, class)
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    # TODO: INVESTIGATE THE COMMENTED OUT PARTS AND DECIDE IF TO ERASE OR UNCOMMENT
    number_of_classes = prediction[0].shape[1] - 5
    candidates_above_thres = prediction[..., 4] > conf_thres

    # Settings
    # min_box_width_and_height = 2
    max_box_width_and_height = 4096
    max_num_of_detections = 300
    require_redundant_detections = True
    multi_label_per_box = number_of_classes > 1  # (adds 0.5ms/img)
    output = [None] * prediction.shape[0]
    for image_idx, pred in enumerate(prediction):
        # Apply constraints
        # pred[((pred[..., 2:4] < min_box_width_and_height) | (pred[..., 2:4] > max_box_width_and_height)).any(1), 4] = 0  # width-height
        pred = pred[candidates_above_thres[image_idx]]  # confidence

        # If none remain process next image
        if not pred.shape[0]:
            continue

        # Compute confidence = object_conf * class_conf
        pred[:, 5:] *= pred[:, 4:5]
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = convert_xywh_bbox_to_xyxy(pred[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label_per_box:
            i, j = (pred[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            pred = torch.cat((box[i], pred[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = pred[:, 5:].max(1, keepdim=True)
            pred = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            pred = pred[(pred[:, 5:6] == torch.tensor(classes, device=pred.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        number_of_boxes = pred.shape[0]
        if not number_of_boxes:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        # CREATE AN OFFSET OF THE PREDICTIVE BOX OF DIFFERENT CLASSES IF not agnostic
        offset = pred[:, 5:6] * (0 if agnostic else max_box_width_and_height)
        boxes, scores = pred[:, :4] + offset, pred[:, 4]
        idx_to_keep = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if idx_to_keep.shape[0] > max_num_of_detections:  # limit number of detections
            idx_to_keep = idx_to_keep[:max_num_of_detections]
        if merge and (1 < number_of_boxes < 3000):
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[idx_to_keep], boxes) > iou_thres  # iou matrix
                box_weights = iou * scores[None]
                # MERGED BOXES
                pred[idx_to_keep, :4] = torch.mm(box_weights, pred[:, :4]).float() / box_weights.sum(1, keepdim=True)
                if require_redundant_detections:
                    idx_to_keep = idx_to_keep[iou.sum(1) > 1]
            except RuntimeError:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(pred, idx_to_keep, pred.shape, idx_to_keep.shape)
                pass

        output[image_idx] = pred[idx_to_keep]

    return output


def check_img_size_divisibilty(img_size: int, stride: int = 32):
    """
    :param img_size: Int, the size of the image (H or W).
    :param stride: Int, the number to check if img_size is divisible by.
    :return: (True, None) if img_size is divisble by stride, (False, Suggestions) if it's not.
        Note: Suggestions are the two closest numbers to img_size that *are* divisible by stride.
        For example if img_size=321, stride=32, it will return (False,(352, 320)).
    """
    new_size = make_divisible(img_size, int(stride))
    if new_size != img_size:
        return False, (new_size, make_divisible(img_size, int(stride), ceil=False))
    else:
        return True, None


def make_divisible(x, divisor, ceil=True):
    """
    Returns x evenly divisible by divisor.
    If ceil=True it will return the closest larger number to the original x, and ceil=False the closest smaller number.
    """
    if ceil:
        return math.ceil(x / divisor) * divisor
    else:
        return math.floor(x / divisor) * divisor


def matrix_non_max_suppression(pred, conf_thres: float = 0.1, kernel: str = 'gaussian',
                               sigma: float = 3.0, max_num_of_detections: int = 500):
    """Performs Matrix Non-Maximum Suppression (NMS) on inference results
        https://arxiv.org/pdf/1912.04488.pdf
        :param pred: raw model prediction (in test mode) - a Tensor of shape [batch, num_predictions, 85]
        where each item format is (x, y, w, h, object_conf, class_conf, ... 80 classes score ...)
        :param conf_thres: below the confidence threshold - prediction are discarded
        :param kernel: type of kernel to use ['gaussian', 'linear']
        :param sigma: sigma for the gussian kernel
        :param max_num_of_detections: maximum number of boxes to output
        :return:  list of (x1, y1, x2, y2, object_conf, class_conf, class)

    Returns:
         detections list with shape: (x1, y1, x2, y2, conf, cls)
    """
    # MULTIPLY CONF BY CLASS CONF TO GET COMBINED CONFIDENCE
    class_conf, class_pred = pred[:, :, 5:].max(2)
    pred[:, :, 4] *= class_conf

    # BOX (CENTER X, CENTER Y, WIDTH, HEIGHT) TO (X1, Y1, X2, Y2)
    pred[:, :, :4] = convert_xywh_bbox_to_xyxy(pred[:, :, :4])

    # DETECTIONS ORDERED AS (x1y1x2y2, obj_conf, class_conf, class_pred)
    pred = torch.cat((pred[:, :, :5], class_pred.unsqueeze(2)), 2)

    # SORT DETECTIONS BY DECREASING CONFIDENCE SCORES
    sort_ind = (-pred[:, :, 4]).argsort()
    pred = torch.stack([pred[i, sort_ind[i]] for i in range(pred.shape[0])])[:, 0:max_num_of_detections]

    ious = calc_bbox_iou_matrix(pred)

    ious = ious.triu(1)

    # CREATE A LABELS MASK, WE WANT ONLY BOXES WITH THE SAME LABEL TO AFFECT EACH OTHER
    labels = pred[:, :, 5:]
    labeles_matrix = (labels == labels.transpose(2, 1)).float().triu(1)

    ious *= labeles_matrix
    ious_cmax, _ = ious.max(1)
    ious_cmax = ious_cmax.unsqueeze(2).repeat(1, 1, max_num_of_detections)

    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (ious ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (ious_cmax ** 2))
        decay, _ = (decay_matrix / compensate_matrix).min(dim=1)
    else:
        decay = (1 - ious) / (1 - ious_cmax)
        decay, _ = decay.min(dim=1)

    pred[:, :, 4] *= decay

    output = [pred[i, pred[i, :, 4] > conf_thres] for i in range(pred.shape[0])]

    return output


class NMS_Type(str, Enum):
    """
    Type of non max suppression algorithm that can be used for post processing detection
    """
    ITERATIVE = 'iterative'
    MATRIX = 'matrix'


def calc_batch_prediction_accuracy(output: torch.Tensor, targets: torch.Tensor, height: int, width: int,  # noqa: C901
                                   iou_thres: IouThreshold) -> tuple:
    """

    :param output:       list (of length batch_size) of Tensors of shape (num_detections, 6)
                         format:     (x1, y1, x2, y2, confidence, class_label) where x1,y1,x2,y2 are according to image size
    :param targets:      targets for all images of shape (total_num_targets, 6)
                         format:     (image_index, x, y, w, h, label) where x,y,w,h are in range [0,1]
    :param height,width: dimensions of the image
    :param iou_thres:    Threshold to compute the mAP
    :param device:       'cuda'\'cpu' - where the computations are made
    :return:
    """
    batch_metrics = []
    batch_images_counter = 0
    device = targets.device

    if not iou_thres.is_range():
        num_ious = 1
        ious = torch.tensor([iou_thres[0]]).to(device)
    else:
        num_ious = int(round((iou_thres[1] - iou_thres[0]) / 0.05)) + 1
        ious = torch.linspace(iou_thres[0], iou_thres[1], num_ious).to(device)

    for i, pred in enumerate(output):
        labels = targets[targets[:, 0] == i, 1:]
        labels_num = len(labels)
        target_class = labels[:, 0].tolist() if labels_num else []
        batch_images_counter += 1

        if pred is None:
            if labels_num:
                batch_metrics.append(
                    (np.zeros((0, num_ious), dtype=np.bool), np.array([], dtype=np.float32), np.array([], dtype=np.float32), target_class))
            continue

        # CHANGE bboxes TO FIT THE IMAGE SIZE
        change_bbox_bounds_for_image_size(pred, (height, width))

        # ZEROING ALL OF THE bbox PREDICTIONS BEFORE MAX IOU FILTERATION
        correct = torch.zeros(len(pred), num_ious, dtype=torch.bool, device=device)
        if labels_num:
            detected = []
            tcls_tensor = labels[:, 0]

            target_bboxes = convert_xywh_bbox_to_xyxy(labels[:, 1:5])
            target_bboxes[:, [0, 2]] *= width
            target_bboxes[:, [1, 3]] *= height

            # SEARCH FOR CORRECT PREDICTIONS
            # Per target class
            for cls in torch.unique(tcls_tensor):
                target_index = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                pred_index = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                # Search for detections
                if pred_index.shape[0]:
                    # Prediction to target ious
                    iou, i = box_iou(pred[pred_index, :4], target_bboxes[target_index]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (iou > ious[0]).nonzero(as_tuple=False):
                        detected_target = target_index[i[j]]
                        if detected_target.item() not in detected_set:
                            detected_set.add(detected_target.item())
                            detected.append(detected_target)
                            correct[pred_index[j]] = iou[j] > ious  # iou_thres is 1xn
                            if len(detected) == labels_num:  # all targets already located in image
                                break

        # APPEND STATISTICS (CORRECT, CONF, PCLS, TCLS)
        batch_metrics.append((correct.cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, -1].cpu().numpy(), target_class))

    return batch_metrics, batch_images_counter


class AnchorGenerator:
    logger = get_logger(__name__)

    @staticmethod
    def _metric(objects, anchors):
        """ measure how 'far' each object is from the closest anchor
            :returns a matrix n by number of objects and the measurements to the closest anchor for each object
        """
        r = objects[:, None] / anchors[None]
        matrix = np.amin(np.minimum(r, 1. / r), axis=2)
        return matrix, matrix.max(1)

    @staticmethod
    def _anchor_fitness(objects, anchors, thresh):
        """ how well the anchors fit the objects"""
        _, best = AnchorGenerator._metric(objects, anchors)
        return (best * (best > thresh)).mean()  # fitness

    @staticmethod
    def _print_results(objects, anchors, thresh, num_anchors, img_size):
        # SORT SMALL TO LARGE (BY AREA)
        anchors = anchors[np.argsort(anchors.prod(1))]
        x, best = AnchorGenerator._metric(objects, anchors)
        best_possible_recall = (best > thresh).mean()
        anchors_above_thesh = (x > thresh).mean() * num_anchors

        AnchorGenerator.logger.info(
            f'thr={thresh:.2f}: {best_possible_recall:.4f} best possible recall, {anchors_above_thesh:.2f} anchors past thr')
        AnchorGenerator.logger.info(f'num_anchors={num_anchors}, img_size={img_size}')
        AnchorGenerator.logger.info(
            f' metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, past_thr={x[x > thresh].mean():.3f}-mean: ')

        for i, mean in enumerate(anchors):
            print('%i,%i' % (round(mean[0]), round(mean[1])),
                  end=',  ' if i < len(anchors) - 1 else '\n')  # use in *.cfg

    @staticmethod
    def _plot_object_distribution(objects, anchors):
        selected = np.random.choice(objects.shape[0], size=objects.shape[0] // 50, replace=False)

        distance_matrix = np.sqrt(np.power(objects[:, :, None] - anchors[:, :, None].T, 2).sum(1))
        labels = np.argmin(distance_matrix, axis=1)
        plt.scatter(objects[selected, 0], objects[selected, 1], c=labels[selected], marker='.')
        plt.scatter(anchors[:, 0], anchors[:, 1], marker='P')
        plt.show()

    @staticmethod
    def _generate_anchors(dataset, num_anchors=9, thresh=0.25, gen=1000):
        """ Creates kmeans-evolved anchors from training dataset
            Based on the implementation by Ultralytics for Yolo V5

            :param dataset: a loaded dataset (must be with cached labels and "train_sample_loading_method":'rectangular')
            :param num_anchors: number of anchors
            :param thresh: anchor-label wh ratio threshold used to asses if a label can be detected by an anchor.
                    it means that the aspect ratio of the object is not more than thres from the aspect ratio of the anchor.
            :param gen: generations to evolve anchors using genetic algorithm. after kmeans, this algorithm iteratively
                    make minor random changes in the anchors and if a change imporve the anchors-data fit it evolves the
                    anchors.
            :returns anchors array num_anchors by 2 (x,y) normalized to image size
        """
        _prefix = 'Anchors Generator: '
        img_size = dataset.img_size
        assert dataset.cache_labels, "dataset labels have to be cached before generating anchors"

        image_shapes = np.array(
            [dataset.exif_size(Image.open(f)) for f in tqdm(dataset.img_files, desc='Reading image shapes')])

        # Get label wh
        shapes = img_size * image_shapes / image_shapes.max(1, keepdims=True)
        objects_wh = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])

        # Filter
        i = (objects_wh < 3.0).any(1).sum()
        if i:
            AnchorGenerator.logger.warning(
                f'Extremely small objects found. {i} of {len(objects_wh)} labels are < 3 pixels in size.')
        object_wh_filtered = objects_wh[(objects_wh >= 2.0).any(1)]

        # Kmeans calculation
        AnchorGenerator.logger.info(f'Running kmeans for {num_anchors} anchors on {len(object_wh_filtered)} points...')
        mean_wh = object_wh_filtered.std(0)  # sigmas for whitening
        anchors, dist = kmeans(object_wh_filtered / mean_wh, num_anchors, iter=30)  # points, mean distance
        # MEANS WHERE NORMALIZED. SCALE THEM BACK TO IMAGE SIZE
        anchors *= mean_wh

        AnchorGenerator.logger.info('Initial results')
        AnchorGenerator._print_results(objects_wh, anchors, thresh, num_anchors, img_size)
        AnchorGenerator._plot_object_distribution(objects_wh, anchors)

        # EVOLVE
        fitness, generations, mutation_prob, sigma = AnchorGenerator._anchor_fitness(object_wh_filtered, anchors,
                                                                                     thresh), anchors.shape, 0.9, 0.1
        progress_bar = tqdm(range(gen), desc=f'{_prefix}Evolving anchors with Genetic Algorithm:')
        for _ in progress_bar:
            v = np.ones(generations)
            while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
                v = ((np.random.random(generations) < mutation_prob) * np.random.random() * np.random.randn(
                    *generations) * sigma + 1).clip(0.3, 3.0)
            evolved_anchors = (anchors * v).clip(min=2.0)
            evolved_anchors_fitness = AnchorGenerator._anchor_fitness(object_wh_filtered, evolved_anchors, thresh)
            if evolved_anchors_fitness > fitness:
                fitness, anchors = evolved_anchors_fitness, evolved_anchors.copy()
                progress_bar.desc = f'{_prefix}Evolving anchors with Genetic Algorithm: fitness = {fitness:.4f}'

        AnchorGenerator.logger.info('Final results')
        AnchorGenerator._print_results(objects_wh, anchors, thresh, num_anchors, img_size)
        AnchorGenerator._plot_object_distribution(objects_wh, anchors)

        anchors = anchors[np.argsort(anchors.prod(1))]
        anchors_list = np.round(anchors.reshape((3, -1))).astype(np.int32).tolist()
        return anchors_list

    @staticmethod
    def __call__(dataset, num_anchors=9, thresh=0.25, gen=1000):
        return AnchorGenerator._generate_anchors(dataset, num_anchors, thresh, gen)


def plot_coco_datasaet_images_with_detections(data_loader, num_images_to_plot=1):
    """
    plot_coco_images
        :param data_loader:
        :param num_images_to_plot:
        :return:
    # """
    images_counter = 0

    # PLOT ONE image AND ONE GROUND_TRUTH bbox
    for imgs, targets in data_loader:

        # PLOTS TRAINING IMAGES OVERLAID WITH TARGETS
        imgs = imgs.cpu().numpy()
        targets = targets.cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        batch_size, _, h, w = imgs.shape

        # LIMIT PLOT TO 16 IMAGES
        batch_size = min(batch_size, 16)

        # NUMBER OF SUBPLOTS
        ns = np.ceil(batch_size ** 0.5)

        for i in range(batch_size):
            boxes = convert_xywh_bbox_to_xyxy(torch.from_numpy(targets[targets[:, 0] == i, 2:6])).cpu().detach().numpy().T
            boxes[[0, 2]] *= w
            boxes[[1, 3]] *= h
            plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
            plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
            plt.axis('off')
        fig.tight_layout()
        plt.show()
        plt.close()

        images_counter += 1
        if images_counter == num_images_to_plot:
            break


def undo_image_preprocessing(im_tensor: torch.Tensor) -> np.ndarray:
    """
    :param im_tensor: images in a batch after preprocessing for inference, RGB, (B, C, H, W)
    :return:          images in a batch in cv2 format, BGR, (B, H, W, C)
    """
    im_np = im_tensor.cpu().numpy()
    im_np = im_np[:, ::-1, :, :].transpose(0, 2, 3, 1)
    im_np *= 255.
    return np.ascontiguousarray(im_np, dtype=np.uint8)


class DetectionVisualization:
    @staticmethod
    def _generate_color_mapping(num_classes: int) -> List[Tuple[int]]:
        """
        Generate a unique BGR color for each class
        """
        cmap = plt.cm.get_cmap('gist_rainbow', num_classes)
        colors = [cmap(i, bytes=True)[:3][::-1] for i in range(num_classes)]
        return [tuple(int(v) for v in c) for c in colors]

    @staticmethod
    def _draw_box_title(color_mapping: List[Tuple[int]], class_names: List[str], box_thickness: int,
                        image_np: np.ndarray, x1: int, y1: int, x2: int, y2: int, class_id: int,
                        pred_conf: float = None):
        color = color_mapping[class_id]
        class_name = class_names[class_id]

        # Draw the box
        image_np = cv2.rectangle(image_np, (x1, y1), (x2, y2), color, box_thickness)

        # Caption with class name and confidence if given
        text_color = (255, 255, 255)  # white
        title = f'{class_name}  {str(round(pred_conf, 2)) if pred_conf is not None else ""}'
        image_np = cv2.rectangle(image_np, (x1, y1 - 15), (x1 + len(title) * 10, y1), color, cv2.FILLED)
        image_np = cv2.putText(image_np, title, (x1, y1 - box_thickness), 2, .5, text_color, 1, lineType=cv2.LINE_AA)

        return image_np

    @staticmethod
    def _visualize_image(image_np: np.ndarray, pred_boxes: np.ndarray, target_boxes: np.ndarray,
                         class_names: List[str], box_thickness: int, gt_alpha: float, image_scale: float,
                         checkpoint_dir: str, image_name: str):
        image_np = cv2.resize(image_np, (0, 0), fx=image_scale, fy=image_scale, interpolation=cv2.INTER_NEAREST)
        color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

        # Draw predictions
        pred_boxes[:, :4] *= image_scale
        for box in pred_boxes:
            image_np = DetectionVisualization._draw_box_title(color_mapping, class_names, box_thickness,
                                                              image_np, *box[:4].astype(int),
                                                              class_id=int(box[5]), pred_conf=box[4])

        # Draw ground truths
        target_boxes_image = np.zeros_like(image_np, np.uint8)
        for box in target_boxes:
            target_boxes_image = DetectionVisualization._draw_box_title(color_mapping, class_names, box_thickness,
                                                                        target_boxes_image, *box[2:],
                                                                        class_id=box[1])

        # Transparent overlay of ground truth boxes
        mask = target_boxes_image.astype(bool)
        image_np[mask] = cv2.addWeighted(image_np, 1 - gt_alpha, target_boxes_image, gt_alpha, 0)[mask]

        if checkpoint_dir is None:
            return image_np
        else:
            cv2.imwrite(os.path.join(checkpoint_dir, str(image_name) + '.jpg'), image_np)

    @staticmethod
    def _scaled_ccwh_to_xyxy(target_boxes: np.ndarray, h: int, w: int, image_scale: float) -> np.ndarray:
        """
        Modifies target_boxes inplace
        :param target_boxes:    (c1, c2, w, h) boxes in [0, 1] range
        :param h:               image height
        :param w:               image width
        :param image_scale:     desired scale for the boxes w.r.t. w and h
        :return:                targets in (x1, y1, x2, y2) format
                                in range [0, w * self.image_scale] [0, h * self.image_scale]
        """
        # unscale
        target_boxes[:, 2:] *= np.array([[w, h, w, h]])

        # x1 = c1 - w // 2; y1 = c2 - h // 2
        target_boxes[:, 2] -= target_boxes[:, 4] // 2
        target_boxes[:, 3] -= target_boxes[:, 5] // 2
        # x2 = w + x1; y2 = h + y1
        target_boxes[:, 4] += target_boxes[:, 2]
        target_boxes[:, 5] += target_boxes[:, 3]

        target_boxes[:, 2:] *= image_scale
        target_boxes = target_boxes.astype(int)
        return target_boxes

    @staticmethod
    def visualize_batch(image_tensor: torch.Tensor, pred_boxes: List[torch.Tensor], target_boxes: torch.Tensor,
                        batch_name: Union[int, str], class_names: List[str], checkpoint_dir: str = None,
                        undo_preprocessing_func: Callable[[torch.Tensor], np.ndarray] = undo_image_preprocessing,
                        box_thickness: int = 2, image_scale: float = 1., gt_alpha: float = .4):
        """
        A helper function to visualize detections predicted by a network:
        saves images into a given path with a name that is {batch_name}_{imade_idx_in_the_batch}.jpg, one batch per call.
        Colors are generated on the fly: uniformly sampled from color wheel to support all given classes.

        Adjustable:
            * Ground truth box transparency;
            * Box width;
            * Image size (larger or smaller than what's provided)

        :param image_tensor:            rgb images, (B, H, W, 3)
        :param pred_boxes:              boxes after NMS for each image in a batch, each (Num_boxes, 6),
                                        values on dim 1 are: x1, y1, x2, y2, confidence, class
        :param target_boxes:            (Num_targets, 6), values on dim 1 are: image id in a batch, class, x y w h
                                        (coordinates scaled to [0, 1])
        :param batch_name:              id of the current batch to use for image naming

        :param class_names:             names of all classes, each on its own index
        :param checkpoint_dir:          a path where images with boxes will be saved. if None, the result images will
                                        be returns as a list of numpy image arrays

        :param undo_preprocessing_func: a function to convert preprocessed images tensor into a batch of cv2-like images
        :param box_thickness:           box line thickness in px
        :param image_scale:             scale of an image w.r.t. given image size,
                                        e.g. incoming images are (320x320), use scale = 2. to preview in (640x640)
        :param gt_alpha:                a value in [0., 1.] transparency on ground truth boxes,
                                        0 for invisible, 1 for fully opaque
        """
        image_np = undo_preprocessing_func(image_tensor.detach())
        targets = DetectionVisualization._scaled_ccwh_to_xyxy(target_boxes.detach().cpu().numpy(), *image_np.shape[1:3],
                                                              image_scale)

        out_images = []
        for i in range(image_np.shape[0]):
            preds = pred_boxes[i].detach().cpu().numpy() if pred_boxes[i] is not None else np.empty((0, 6))
            targets_cur = targets[targets[:, 0] == i]

            image_name = '_'.join([str(batch_name), str(i)])
            res_image = DetectionVisualization._visualize_image(image_np[i], preds, targets_cur, class_names, box_thickness, gt_alpha, image_scale, checkpoint_dir, image_name)
            if res_image is not None:
                out_images.append(res_image)

        return out_images


class Anchors(nn.Module):
    """
    A wrapper function to hold the anchors used by detection models such as Yolo
    """

    def __init__(self, anchors_list: List[List], strides: List[int]):
        """
        :param anchors_list: of the shape [[w1,h1,w2,h2,w3,h3], [w4,h4,w5,h5,w6,h6] .... where each sublist holds
            the width and height of the anchors of a specific detection layer.
            i.e. for a model with 3 detection layers, each containing 5 anchors the format will be a of 3 sublists of 10 numbers each
            The width and height are in pixels (not relative to image size)
        :param strides: a list containing the stride of the layers from which the detection heads are fed.
            i.e. if the firs detection head is connected to the backbone after the input dimensions were reduces by 8, the first number will be 8
        """
        super().__init__()

        self.__anchors_list = anchors_list
        self.__strides = strides

        self._check_all_lists(anchors_list)
        self._check_all_len_equal_and_even(anchors_list)

        self._stride = nn.Parameter(torch.Tensor(strides).float(), requires_grad=False)
        anchors = torch.Tensor(anchors_list).float().view(len(anchors_list), -1, 2)
        self._anchors = nn.Parameter(anchors / self._stride.view(-1, 1, 1), requires_grad=False)
        self._anchor_grid = nn.Parameter(anchors.clone().view(len(anchors_list), 1, -1, 1, 1, 2), requires_grad=False)

    @staticmethod
    def _check_all_lists(anchors: list) -> bool:
        for a in anchors:
            if not isinstance(a, (list, ListConfig)):
                raise RuntimeError('All objects of anchors_list must be lists')

    @staticmethod
    def _check_all_len_equal_and_even(anchors: list) -> bool:
        len_of_first = len(anchors[0])
        for a in anchors:
            if len(a) % 2 == 1 or len(a) != len_of_first:
                raise RuntimeError('All objects of anchors_list must be of the same even length')

    @property
    def stride(self) -> nn.Parameter:
        return self._stride

    @property
    def anchors(self) -> nn.Parameter:
        return self._anchors

    @property
    def anchor_grid(self) -> nn.Parameter:
        return self._anchor_grid

    @property
    def detection_layers_num(self) -> int:
        return self._anchors.shape[0]

    @property
    def num_anchors(self) -> int:
        return self._anchors.shape[1]

    def __repr__(self):
        return f"anchors_list: {self.__anchors_list} strides: {self.__strides}"
