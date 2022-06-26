from typing import List, Tuple, Optional

import torch
from torchmetrics import Metric

import super_gradients
from super_gradients.training.utils.detection_utils import compute_detection_matching, compute_detection_metrics

from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback, IouThreshold


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
        self.add_state("matching", default=[], dist_reduce_fx=None)

        self.iou_thresholds = iou_thres

    def update(self, preds: List[torch.Tensor], target: torch.Tensor, device: str, inputs: torch.tensor, crowd_gts=None):
        """
        Apply NMS and match all the predictions and targets of a given batch, and update the metric state accordingly.

        :param preds :    list (of length batch_size) of Tensors of shape (num_detections, 6)
                            format:  (x1, y1, x2, y2, confidence, class_label) where x1,y1,x2,y2 non normalized
        :param target:    targets for all images of shape (total_num_targets, 6)
                            format:  (index, x, y, w, h, label) where x,y,w,h are in range [0,1]
        :param device:    Device to run on
        :param inputs:    Input image tensor of shape (batch_size, n_img, height, width)
        :param crowd_gts: crowd targets for all images of shape (total_num_targets, 6)
                          format:  (index, x, y, w, h, label) where x,y,w,h are in range [0,1]
        :return:
        """
        preds = self.post_prediction_callback(preds, device=device)

        _, _, height, width = inputs.shape
        new_matching = compute_detection_matching(
            preds, target, height, width, self.iou_thresholds, crowd_targets=crowd_gts)

        accumulated_matching = getattr(self, "matching")
        setattr(self, "matching", accumulated_matching + new_matching)

    def compute(self):
        accumulated_matching = getattr(self, "matching")
        precision, recall, mean_ap, f1 = -1, -1, -1, -1

        if len(accumulated_matching):
            matching_tensors = [torch.cat(x, 0) for x in list(zip(*accumulated_matching))]

            # shape (n_class, nb_iou_thrs)
            device = matching_tensors[0].device
            precision, recall, ap, f1, unique_classes = compute_detection_metrics(*matching_tensors, device=device)

            precision, recall, f1 = precision[:, 0].mean(), recall[:, 0].mean(), f1[:, 0].mean()
            mean_ap = ap.mean()

        return {"Precision": precision, "Recall": recall, self.map_str: mean_ap, "F1": f1}

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
                metrics += state_dict["matching"]

            setattr(self, "matching", metrics)


# def compute_detection_metrics_from_accumulated_matching(
#         accumulated_matching: List[Tuple[torch.Tensor]]
# ) -> Tuple[float, float, float, float]:
#     """Compute precision, recall, mean_ap and f1 using the matching accumulated through the "update" method.
#     When no matching, return -1 for every metric.
#
#     :param accumulated_matching: List of length (m_images) with Tuples of:
#                                     preds_matched, preds_to_ignore, preds_scores, preds_cls, targets_cls
#
#     :return:
#         :precision:      Average over classes, computed for the smallest IoU threshold
#         :recall:         Average over classes, computed for the smallest IoU threshold
#         :mean_ap:        Average over classes and IuO thresholds
#         :f1:             Average over classes, computed for the smallest IoU threshold
#         :unique_classes: All unique classes present in the input
#     """

    # return precision, recall, mean_ap, f1, unique_classes
