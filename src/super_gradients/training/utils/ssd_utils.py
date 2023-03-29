import itertools
from math import sqrt
from typing import List

import numpy as np
import torch

from super_gradients.training.utils.detection_utils import non_max_suppression, NMS_Type, matrix_non_max_suppression, DetectionPostPredictionCallback


class DefaultBoxes(object):
    """
    Default Boxes, (aka: anchor boxes or priors boxes) used by SSD model
    """

    def __init__(self, fig_size: int, feat_size: List[int], scales: List[int], aspect_ratios: List[List[int]], scale_xy=0.1, scale_wh=0.2):
        """
        For each feature map i (each predicting level, grids) the anchors (a.k.a. default boxes) will be:
        [
            [s, s], [sqrt(s * s_next), sqrt(s * s_next)],
            [s * sqrt(alpha1), s / sqrt(alpha1)], [s / sqrt(alpha1), s * sqrt(alpha1)],
            ...
            [s * sqrt(alphaN), s / sqrt(alphaN)], [s / sqrt(alphaN), s * sqrt(alphaN)]
        ] / fig_size
        where:
            * s = scale[i] - this level's scale
            * s_next = scale[i + 1] - next level's scale
            * alpha1, ... alphaN - this level's alphas, e.g. [2, 3]
            * fig_size - input image resolution

        Because of division by image resolution, the anchors will be in image coordinates normalized to [0, 1]

        :param fig_size:        input image resolution
        :param feat_size:       resolution of all feature maps with predictions (grids)
        :param scales:          anchor sizes in pixels for each feature level;
                                one value per level will be used to generate anchors based on the formula above
        :param aspect_ratios:   lists of alpha values for each feature map
        :param scale_xy:        predicted boxes will be with a factor scale_xy
                                so will be multiplied by scale_xy during post-prediction processing;
                                e.g. scale 0.1 means that prediction will be 10 times bigger
                                (improves predictions quality)
        :param scale_wh:        same logic as in scale_xy, but for width and height.
        """
        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        self.num_anchors = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx]
            sk2 = scales[idx + 1]
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            all_sizes = np.array(all_sizes) / fig_size
            self.num_anchors.append(len(all_sizes))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / sfeat, (i + 0.5) / sfeat
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


class SSDPostPredictCallback(DetectionPostPredictionCallback):
    """
    post prediction callback module to convert and filter predictions coming from the SSD net to a format
    used by all other detection models
    """

    def __init__(
        self,
        conf: float = 0.001,
        iou: float = 0.6,
        classes: list = None,
        max_predictions: int = 300,
        nms_type: NMS_Type = NMS_Type.ITERATIVE,
        multi_label_per_box=True,
    ):
        """
        Predictions of SSD contain unnormalized probabilities for a background class,
        together with confidences for all the dataset classes. Background will be utilized and discarded,
        so this callback will return 0-based classes without background
        :param conf: confidence threshold
        :param iou: IoU threshold
        :param classes: (optional list) filter by class
        :param nms_type: the type of nms to use (iterative or matrix)
        :param multi_label_per_box: whether to use re-use each box with all possible labels
                                    (instead of the maximum confidence all confidences above threshold
                                    will be sent to NMS)
        """
        super(SSDPostPredictCallback, self).__init__()
        self.conf = conf
        self.iou = iou
        self.nms_type = nms_type
        self.classes = classes
        self.max_predictions = max_predictions

        self.multi_label_per_box = multi_label_per_box

    def forward(self, predictions, device=None):
        nms_input = predictions[0]
        if self.nms_type == NMS_Type.ITERATIVE:
            nms_res = non_max_suppression(
                nms_input, conf_thres=self.conf, iou_thres=self.iou, multi_label_per_box=self.multi_label_per_box, with_confidence=True
            )
        else:
            nms_res = matrix_non_max_suppression(nms_input, conf_thres=self.conf, max_num_of_detections=self.max_predictions)

        return self._filter_max_predictions(nms_res)

    def _filter_max_predictions(self, res: List) -> List:
        res[:] = [im[: self.max_predictions] if (im is not None and im.shape[0] > self.max_predictions) else im for im in res]
        return res
