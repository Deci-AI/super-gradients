from typing import Dict, Optional, Union

import numpy as np
import torch
from torchmetrics import Metric

import super_gradients
from super_gradients.training.utils import tensor_container_to_device
from super_gradients.training.utils.detection_utils import compute_detection_matching, compute_detection_metrics,\
    calc_batch_prediction_accuracy
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback, IouThreshold
from super_gradients.common.abstractions.abstract_logger import get_logger
logger = get_logger(__name__)


def compute_ap(recall, precision, method: str = 'interp'):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        :param recall:    The recall curve - ndarray [1, points in curve]
        :param precision: The precision curve - ndarray [1, points in curve]
        :param method: 'continuous', 'interp'
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # IN ORDER TO CALCULATE, WE HAVE TO MAKE SURE THE CURVES GO ALL THE WAY TO THE AXES (FROM X=0 TO Y=0)
    # THIS IS HOW IT IS COMPUTED IN  ORIGINAL REPO - A MORE CORRECT COMPUTE WOULD BE ([0.], recall, [recall[-1] + 1E-3])
    wrapped_recall = np.concatenate(([0.], recall, [1.0]))
    wrapped_precision = np.concatenate(([1.], precision, [0.]))

    # COMPUTE THE PRECISION ENVELOPE
    wrapped_precision = np.flip(np.maximum.accumulate(np.flip(wrapped_precision)))

    # INTEGRATE AREA UNDER CURVE
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, wrapped_recall, wrapped_precision), x)  # integrate
    else:  # 'continuous'
        i = np.where(wrapped_recall[1:] != wrapped_recall[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((wrapped_recall[i + 1] - wrapped_recall[i]) * wrapped_precision[i + 1])  # area under curve

    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # SORT BY OBJECTNESS
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # FIND UNIQUE CLASSES
    unique_classes = np.unique(target_cls)

    # CREATE PRECISION-RECALL CURVE AND COMPUTE AP FOR EACH CLASS
    pr_score = 0.1  # SCORE TO EVALUATE P AND R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # NUMBER CLASS, NUMBER IOU THRESHOLDS (I.E. 10 FOR MAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        ground_truth_num = (target_cls == c).sum()  # NUMBER OF GROUND TRUTH OBJECTS
        predictions_num = i.sum()  # NUMBER OF PREDICTED OBJECTS

        if predictions_num == 0 or ground_truth_num == 0:
            continue
        else:
            # ACCUMULATE FPS AND TPS
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # RECALL
            recall = tpc / (ground_truth_num + 1e-16)  # RECALL CURVE
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # R AT PR_SCORE, NEGATIVE X, XP BECAUSE XP DECREASES

            # PRECISION
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # P AT PR_SCORE

            # AP FROM RECALL-PRECISION CURVE
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # COMPUTE F1 SCORE (HARMONIC MEAN OF PRECISION AND RECALL)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


class DetectionMetrics(Metric):
    """
    DetectionMetrics

    Metric class for computing F1, Precision, Recall and Mean Average Precision.

    Attributes:

         num_cls: number of classes.

         post_prediction_callback: DetectionPostPredictionCallback to be applied on net's output prior
            to the metric computation (NMS).

         iou_thres: Threshold to compute the mAP (default=IouThreshold.MAP_05_TO_095).

         normalize_targets: Whether to normalize bbox coordinates by image size (default=True).

         dist_sync_on_step: Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. (default=False)
    """
    def __init__(self, num_cls: int,
                 post_prediction_callback: DetectionPostPredictionCallback = None,
                 iou_thres: IouThreshold = IouThreshold.MAP_05_TO_095,
                 normalize_targets: bool = False,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_cls = num_cls
        self.iou_thres = iou_thres
        self.map_str = 'mAP@%.1f' % iou_thres[0] if not iou_thres.is_range() else 'mAP@%.2f:%.2f' % iou_thres
        self.component_names = ["Precision", "Recall", self.map_str, "F1"]
        self.components = len(self.component_names)
        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()
        self.normalize_targets = normalize_targets
        self.world_size = None
        self.rank = None
        self.add_state("metrics", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor, device, inputs, crowd_targets=None):
        if crowd_targets is not None:
            logger.warning('The DatasetInterface was setup to use crowd, but this DetectionMetrics does not handel it.'
                           'If you meant to use crowd, please use DetectionMetricsV2.'
                           'Otherwise set "with_crowd=False" in dataset_params')
        _, _, height, width = inputs.shape
        targets = target.clone()
        if self.normalize_targets:
            targets[:, 2:] /= max(height, width)
        preds = self.post_prediction_callback(preds, device=device)

        metrics, batch_images_counter = calc_batch_prediction_accuracy(preds, targets, height, width,
                                                                       self.iou_thres)
        acc_metrics = getattr(self, "metrics")
        setattr(self, "metrics", acc_metrics + metrics)


    def compute(self):
        precision, recall, f1, mean_precision, mean_recall, mean_ap, mf1 = 0., 0., 0., 0., 0., 0., 0.
        metrics = getattr(self, "metrics")
        metrics = [np.concatenate(x, 0) for x in list(zip(*metrics))]
        if len(metrics):
            precision, recall, average_precision, f1, ap_class = ap_per_class(*metrics)
            if self.iou_thres.is_range():
                precision, recall, average_precision, f1 = precision[:, 0], recall[:, 0], average_precision.mean(
                    1), average_precision[:, 0]

            mean_precision, mean_recall, mean_ap, mf1 = precision.mean(), recall.mean(), average_precision.mean(), f1.mean()

        return {"Precision": mean_precision, "Recall": mean_recall, self.map_str: mean_ap, "F1": mf1}

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        """
        When in distributed mode, stats are aggregated after each forward pass to the metric state. Since these have all
        different sizes we override the synchronization function since it works only for tensors (and use
        all_gather_object)
        @param dist_sync_fn:
        @return:
        """
        if self.world_size is None:
            self.world_size = torch.distributed.get_world_size() if self.is_distributed else -1
        if self.rank is None:
            self.rank = torch.distributed.get_rank() if self.is_distributed else -1

        if self.is_distributed:
            local_state_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
            gathered_state_dicts = [None] * self.world_size
            torch.distributed.barrier()
            torch.distributed.all_gather_object(gathered_state_dicts, local_state_dict)
            metrics = []
            for state_dict in gathered_state_dicts:
                metrics += state_dict["metrics"]
            setattr(self, "metrics", metrics)

class DetectionMetricsV2(Metric):
    """
    DetectionMetrics

    Metric class for computing F1, Precision, Recall and Mean Average Precision.

    Attributes:

         num_cls:                  Number of classes.
         post_prediction_callback: DetectionPostPredictionCallback to be applied on net's output prior
                                   to the metric computation (NMS).
         normalize_targets:        Whether to normalize bbox coordinates by image size (default=False).

         iou_thresholds:    IoU threshold to compute the mAP (default=torch.linspace(0.5, 0.95, 10)).
         recall_thresholds: Recall threshold to compute the mAP (default=torch.linspace(0, 1, 101)).
         score_threshold:   Score threshold to compute Recall, Precision and F1 (default=0.1)
         top_k_predictions: Number of predictions per class used to compute metrics, ordered by confidence score
                            (default=100)

         dist_sync_on_step: Synchronize metric state across processes at each ``forward()``
                            before returning the value at the step. (default=False)
    """
    def __init__(self, num_cls: int,
                 post_prediction_callback: DetectionPostPredictionCallback = None,
                 normalize_targets: bool = False,
                 iou_thres: IouThreshold = IouThreshold.MAP_05_TO_095,
                 recall_thres: torch.Tensor = None,
                 score_thres: float = 0.1,
                 top_k_predictions: int = 100,
                 dist_sync_on_step: bool = False,):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_cls = num_cls
        self.iou_thres = iou_thres
        self.map_str = 'mAP@%.1f' % iou_thres[0] if not iou_thres.is_range() else 'mAP@%.2f:%.2f' % iou_thres
        self.component_names = ["Precision", "Recall", self.map_str, "F1"]
        self.components = len(self.component_names)
        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()
        self.denormalize_targets = not normalize_targets
        self.world_size = None
        self.rank = None
        self.add_state("matching_info", default=[], dist_reduce_fx=None)

        self.iou_thresholds = iou_thres.to_tensor()
        self.recall_thresholds = torch.linspace(0, 1, 101) if recall_thres is None else recall_thres
        self.score_threshold = score_thres
        self.top_k_predictions = top_k_predictions

    def update(self, preds, target: torch.Tensor, device: str,
               inputs: torch.tensor, crowd_targets: Optional[torch.Tensor] = None):
        """
        Apply NMS and match all the predictions and targets of a given batch, and update the metric state accordingly.

        :param preds :        Raw output of the model, the format might change from one model to another, but has to fit
                                the input format of the post_prediction_callback
        :param target:        Targets for all images of shape (total_num_targets, 6)
                                format:  (index, x, y, w, h, label) where x,y,w,h are in range [0,1]
        :param device:        Device to run on
        :param inputs:        Input image tensor of shape (batch_size, n_img, height, width)
        :param crowd_targets: Crowd targets for all images of shape (total_num_targets, 6)
                                 format:  (index, x, y, w, h, label) where x,y,w,h are in range [0,1]
        """
        self.iou_thresholds = self.iou_thresholds.to(device)
        _, _, height, width = inputs.shape

        targets = target.clone()
        crowd_targets = torch.zeros(size=(0, 6), device=device) if crowd_targets is None else crowd_targets.clone()

        preds = self.post_prediction_callback(preds, device=device)
        new_matching_info = compute_detection_matching(
            preds, targets, height, width, self.iou_thresholds, crowd_targets=crowd_targets,
            top_k=self.top_k_predictions, denormalize_targets=self.denormalize_targets)

        accumulated_matching_info = getattr(self, "matching_info")
        setattr(self, "matching_info", accumulated_matching_info + new_matching_info)

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute the metrics for all the accumulated results.
            :return: Metrics of interest
        """
        mean_ap, mean_precision, mean_recall, mean_f1 = -1., -1., -1., -1.
        accumulated_matching_info = getattr(self, "matching_info")

        if len(accumulated_matching_info):
            matching_info_tensors = [torch.cat(x, 0) for x in list(zip(*accumulated_matching_info))]
            self.recall_thresholds = self.recall_thresholds.to(self.device)

            # shape (n_class, nb_iou_thresh)
            ap, precision, recall, f1, unique_classes = compute_detection_metrics(
                *matching_info_tensors, device=self.device, recall_thresholds=self.recall_thresholds,
                score_threshold=self.score_threshold)

            # Precision, recall and f1 are computed for smallest IoU threshold (usually 0.5), averaged over classes
            mean_precision, mean_recall, mean_f1 = precision[:, 0].mean(), recall[:, 0].mean(), f1[:, 0].mean()

            # MaP is averaged over IoU thresholds and over classes
            mean_ap = ap.mean()

        return {"Precision": mean_precision, "Recall": mean_recall, self.map_str: mean_ap, "F1": mean_f1}

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        """
        When in distributed mode, stats are aggregated after each forward pass to the metric state. Since these have all
        different sizes we override the synchronization function since it works only for tensors (and use
        all_gather_object)
        @param dist_sync_fn:
        @return:
        """
        if self.world_size is None:
            self.world_size = torch.distributed.get_world_size() if self.is_distributed else -1
        if self.rank is None:
            self.rank = torch.distributed.get_rank() if self.is_distributed else -1

        if self.is_distributed:
            local_state_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
            gathered_state_dicts = [None] * self.world_size
            torch.distributed.barrier()
            torch.distributed.all_gather_object(gathered_state_dicts, local_state_dict)
            matching_info = []
            for state_dict in gathered_state_dicts:
                matching_info += state_dict["matching_info"]

            matching_info = tensor_container_to_device(matching_info, device=self.device)
            setattr(self, "matching_info", matching_info)
