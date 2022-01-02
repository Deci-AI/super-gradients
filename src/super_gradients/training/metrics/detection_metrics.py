import numpy as np
import torch
from torchmetrics import Metric
from super_gradients.training.utils.detection_utils import calc_batch_prediction_accuracy, DetectionPostPredictionCallback, \
    IouThreshold
import super_gradients


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
    def __init__(self, num_cls,
                 post_prediction_callback: DetectionPostPredictionCallback = None,
                 iou_thres: IouThreshold = IouThreshold.MAP_05_TO_095,
                 dist_sync_on_step=False):
        """


        @param post_prediction_callback:
        @param iou_thres:
        @param dist_sync_on_step:


        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_cls = num_cls
        self.iou_thres = iou_thres
        self.map_str = 'mAP@%.1f' % iou_thres[0] if not iou_thres.is_range() else 'mAP@%.2f:%.2f' % iou_thres
        self.component_names = ["Precision", "Recall", self.map_str, "F1"]
        self.components = len(self.component_names)
        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()

        self.world_size = None
        self.rank = None
        self.add_state("metrics", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor, device, inputs):
        preds = self.post_prediction_callback(preds, device=device)

        _, _, height, width = inputs.shape

        metrics, batch_images_counter = calc_batch_prediction_accuracy(preds, target, height, width,
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
