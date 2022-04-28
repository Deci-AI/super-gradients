import math
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Union, Tuple

import cv2
from scipy.cluster.vq import kmeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import torch
import numpy as np
from torch import nn
from super_gradients.common.abstractions.abstract_logger import get_logger


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


class IouThreshold(tuple, Enum):
    MAP_05 = (0.5, 0.5)
    MAP_05_TO_095 = (0.5, 0.95)

    def is_range(self):
        return self[0] != self[1]


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


