import itertools
from math import sqrt
import numpy as np
import torch
from torch.nn import functional as F

from super_gradients.training.utils.detection_utils import non_max_suppression, NMS_Type, \
    matrix_non_max_suppression, DetectionPostPredictionCallback


class DefaultBoxes(object):
    """
    Default Boxes, (aka: anchor boxes or priors boxes) used by SSD model
    """

    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)

        # For IoU calculation
        self.dboxes_xyxy = self.dboxes.clone()
        self.dboxes_xyxy[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_xyxy[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_xyxy[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_xyxy[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="xyxy"):
        if order == "xyxy":
            return self.dboxes_xyxy
        if order == "xywh":
            return self.dboxes

    @staticmethod
    def dboxes300_coco():
        figsize = 300
        feat_size = [38, 19, 10, 5, 3, 1]
        steps = [8, 16, 32, 64, 100, 300]
        # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
        scales = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        return DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)

    @staticmethod
    def dboxes300_coco_from19():
        """
        This dbox configuration is a bit different from the original dboxes300_coco
        It is suitable for a network taking the first skip connection from a 19x19 layer (instead of 38x38 in the
        original paper).
        This offers less coverage for small objects but more aspect ratios options to larger objects (the original
        paper supports object starting from size 21 pixels, while this config support objects starting from 60 pixels)
        """

        # https://github.com/qfgaohao/pytorch-ssd/blob/f61ab424d09bf3d4bb3925693579ac0a92541b0d/vision/ssd/config/mobilenetv1_ssd_config.py
        figsize = 300
        feat_size = [19, 10, 5, 3, 2, 1]
        steps = [16, 32, 64, 100, 150, 300]
        scales = [60, 105, 150, 195, 240, 285, 330]
        aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
        return DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)

    @staticmethod
    def dboxes256_coco():
        figsize = 256
        feat_size = [32, 16, 8, 4, 2, 1]
        steps = [8, 16, 32, 64, 128, 256]
        # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
        scales = [18, 38, 84, 131, 1177, 223, 269]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        return DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)


class SSDPostPredictCallback(DetectionPostPredictionCallback):
    """
    post prediction callback module to convert and filter predictions coming from the SSD net to a format
    used by all other detection models
    """

    def __init__(self, conf: float = 0.1, iou: float = 0.45, classes: list = None, max_predictions: int = 300,
                 nms_type: NMS_Type = NMS_Type.ITERATIVE,
                 dboxes: DefaultBoxes = DefaultBoxes.dboxes300_coco(), device='cuda'):
        """
        :param conf: confidence threshold
        :param iou: IoU threshold
        :param classes: (optional list) filter by class
        :param nms_type: the type of nms to use (iterative or matrix)
        """
        super(SSDPostPredictCallback, self).__init__()
        self.conf = conf
        self.iou = iou
        self.nms_type = nms_type
        self.classes = classes
        self.max_predictions = max_predictions

        self.dboxes_xywh = dboxes('xywh').to(device)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh
        self.img_size = dboxes.fig_size

    def forward(self, x, device=None):
        bboxes_in = x[0]
        scores_in = x[1]

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] *= self.scale_xy
        bboxes_in[:, :, 2:] *= self.scale_wh

        # CONVERT RELATIVE LOCATIONS INTO ABSOLUTE LOCATION (OUTPUT LOCATIONS ARE RELATIVE TO THE DBOXES)
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, 2:] + self.dboxes_xywh[:, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, 2:]

        scores_in = F.softmax(scores_in, dim=-1)  # TODO softmax without first item?

        # REPLACE THE CONFIDENCE OF CLASS NONE WITH OBJECT CONFIDENCE
        # SSD DOES NOT OUTPUT OBJECT CONFIDENCE, REQUIRED FOR THE NMS
        scores_in[:, :, 0] = torch.max(scores_in[:, :, 1:], dim=2)[0]
        bboxes_in *= self.img_size

        nms_input = torch.cat((bboxes_in, scores_in), dim=2)

        if self.nms_type == NMS_Type.ITERATIVE:
            nms_res = non_max_suppression(nms_input, conf_thres=self.conf, iou_thres=self.iou,
                                          classes=self.classes)
        else:
            nms_res = matrix_non_max_suppression(nms_input, conf_thres=self.conf,
                                                 max_num_of_detections=self.max_predictions)

        # NMS OUTPUT A 0-BASED CLASS LABEL, BUT SSD WORKS WITH 1-BASED CLASS LABEL
        for t in nms_res:
            if t is not None:
                t[:, 5] += 1

        return nms_res
