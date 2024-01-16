import collections
from typing import Dict, Optional, Union, Tuple, List

import numpy as np
import torch
from torchmetrics import Metric

import super_gradients
import super_gradients.common.environment.ddp_utils
from super_gradients.common.object_names import Metrics
from super_gradients.common.registry.registry import register_metric
from super_gradients.training.utils import tensor_container_to_device
from super_gradients.training.utils.detection_utils import (
    compute_detection_matching,
    compute_detection_metrics,
    DistanceMetric,
    EuclideanDistance,
    IoUMatching,
    DistanceMatching,
)
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback, IouThreshold
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


@register_metric(Metrics.DETECTION_METRICS)
class DetectionMetrics(Metric):
    """
    DetectionMetrics

    Metric class for computing F1, Precision, Recall and Mean Average Precision.

    :param num_cls:                         Number of classes.
    :param post_prediction_callback:        DetectionPostPredictionCallback to be applied on net's output prior to the metric computation (NMS).
    :param normalize_targets:               Whether to normalize bbox coordinates by image size.
    :param iou_thres:                       IoU threshold to compute the mAP.
                                            Could be either instance of IouThreshold, a tuple (lower bound, upper_bound) or single scalar.
    :param recall_thres:                    Recall threshold to compute the mAP.
    :param score_thres:                     Score threshold to compute Recall, Precision and F1.
    :param top_k_predictions:               Number of predictions per class used to compute metrics, ordered by confidence score
    :param dist_sync_on_step:               Synchronize metric state across processes at each ``forward()`` before returning the value at the step.
    :param accumulate_on_cpu:               Run on CPU regardless of device used in other parts.
                                            This is to avoid "CUDA out of memory" that might happen on GPU.
    :param calc_best_score_thresholds       Whether to calculate the best score threshold overall and per class
                                            If True, the compute() function will return a metrics dictionary that not
                                            only includes the average metrics calculated across all classes,
                                            but also the optimal score threshold overall and for each individual class.
    :param include_classwise_ap:            Whether to include the class-wise average precision in the returned metrics dictionary.
                                            If enabled, output metrics dictionary will look similar to this:
                                            {
                                                'Precision0.5:0.95': 0.5,
                                                'Recall0.5:0.95': 0.5,
                                                'F10.5:0.95': 0.5,
                                                'mAP0.5:0.95': 0.5,
                                                'AP0.5:0.95_person': 0.5,
                                                'AP0.5:0.95_car': 0.5,
                                                'AP0.5:0.95_bicycle': 0.5,
                                                'AP0.5:0.95_motorcycle': 0.5,
                                                ...
                                            }
                                            Class names are either provided via the class_names parameter or are generated automatically.
    :param class_names:                     Array of class names. When include_classwise_ap=True, will use these names to make
                                            per-class APs keys in the output metrics dictionary.
                                            If None, will use dummy names `class_{idx}` instead.
    :param state_dict_prefix:               A prefix to append to the state dict of the metric. A state dict used to synchronize metric in DDP mode.
                                            It was empirically found that if you have two metric classes A and B(A) that has same state key, for
                                            some reason torchmetrics attempts to sync their states all toghether which causes an error.
                                            In this case adding a prefix to the name of the synchronized state seems to help,
                                            but it is still unclear why it happens.


    """

    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback,
        normalize_targets: bool = False,
        iou_thres: Union[IouThreshold, Tuple[float, float], float] = IouThreshold.MAP_05_TO_095,
        recall_thres: torch.Tensor = None,
        score_thres: float = 0.1,
        top_k_predictions: int = 100,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        calc_best_score_thresholds: bool = True,
        include_classwise_ap: bool = False,
        class_names: List[str] = None,
        state_dict_prefix: str = "",
    ):
        if class_names is None:
            if include_classwise_ap:
                logger.warning(
                    "Parameter 'include_classwise_ap' is set to True, but no class names are provided. "
                    "We will generate dummy class names, but we recommend to provide class names explicitly to"
                    "have meaningful names in reported metrics."
                )
            class_names = ["class_" + str(i) for i in range(num_cls)]
        else:
            class_names = list(class_names)

        if class_names is not None and len(class_names) != num_cls:
            raise ValueError(f"Number of class names ({len(class_names)}) does not match number of classes ({num_cls})")

        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_cls = num_cls
        self.iou_thres = iou_thres
        self.class_names = class_names

        if isinstance(iou_thres, IouThreshold):
            self.iou_thresholds = iou_thres.to_tensor()
        if isinstance(iou_thres, tuple):
            low, high = iou_thres
            self.iou_thresholds = IouThreshold.from_bounds(low, high)
        else:
            self.iou_thresholds = torch.tensor([iou_thres])

        self.map_str = "mAP" + self._get_range_str()
        self.include_classwise_ap = include_classwise_ap

        self.precision_metric_key = f"{state_dict_prefix}Precision{self._get_range_str()}"
        self.recall_metric_key = f"{state_dict_prefix}Recall{self._get_range_str()}"
        self.f1_metric_key = f"{state_dict_prefix}F1{self._get_range_str()}"
        self.map_metric_key = f"{state_dict_prefix}mAP{self._get_range_str()}"

        greater_component_is_better = [
            (self.precision_metric_key, True),
            (self.recall_metric_key, True),
            (self.map_metric_key, True),
            (self.f1_metric_key, True),
        ]

        if self.include_classwise_ap:
            self.per_class_ap_names = [f"{state_dict_prefix}AP{self._get_range_str()}_{class_name}" for class_name in class_names]
            greater_component_is_better += [(key, True) for key in self.per_class_ap_names]

        self.greater_component_is_better = collections.OrderedDict(greater_component_is_better)
        self.component_names = list(self.greater_component_is_better.keys())
        self.calc_best_score_thresholds = calc_best_score_thresholds
        self.best_threshold_per_class_names = [f"Best_score_threshold_{class_name}" for class_name in class_names]

        if self.calc_best_score_thresholds:
            self.component_names.append("Best_score_threshold")

        if self.calc_best_score_thresholds and self.include_classwise_ap:
            self.component_names += self.best_threshold_per_class_names

        self.components = len(self.component_names)

        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()
        self.denormalize_targets = not normalize_targets
        self.world_size = None
        self.rank = None
        self.state_key = f"{state_dict_prefix}matching_info{self._get_range_str()}"
        self.add_state(self.state_key, default=[], dist_reduce_fx=None)

        self.recall_thresholds = torch.linspace(0, 1, 101) if recall_thres is None else torch.tensor(recall_thres, dtype=torch.float32)
        self.score_threshold = score_thres
        self.top_k_predictions = top_k_predictions

        self.accumulate_on_cpu = accumulate_on_cpu

    def update(self, preds, target: torch.Tensor, device: str, inputs: torch.tensor, crowd_targets: Optional[torch.Tensor] = None) -> None:
        """
        Apply NMS and match all the predictions and targets of a given batch, and update the metric state accordingly.

        :param preds:           Raw output of the model, the format might change from one model to another,
                                but has to fit the input format of the post_prediction_callback (cx,cy,wh)
        :param target:          Targets for all images of shape (total_num_targets, 6) LABEL_CXCYWH. format:  (index, label, cx, cy, w, h)
        :param device:          Device to run on
        :param inputs:          Input image tensor of shape (batch_size, n_img, height, width)
        :param crowd_targets:   Crowd targets for all images of shape (total_num_targets, 6), LABEL_CXCYWH
        """
        self.iou_thresholds = self.iou_thresholds.to(device)
        _, _, height, width = inputs.shape
        iou_matcher = IoUMatching(self.iou_thresholds)

        targets = target.clone()
        crowd_targets = torch.zeros(size=(0, 6), device=device) if crowd_targets is None else crowd_targets.clone()

        preds = self.post_prediction_callback(preds, device=device)

        new_matching_info = compute_detection_matching(
            preds,
            targets,
            height,
            width,
            iou_thresholds=iou_matcher.get_thresholds(),
            matching_strategy=iou_matcher,
            crowd_targets=crowd_targets,
            top_k=self.top_k_predictions,
            denormalize_targets=self.denormalize_targets,
            device=self.device,
            return_on_cpu=self.accumulate_on_cpu,
        )

        accumulated_matching_info = getattr(self, self.state_key)
        setattr(self, self.state_key, accumulated_matching_info + new_matching_info)

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute the metrics for all the accumulated results.
        :return: Metrics of interest
        """
        mean_ap, mean_precision, mean_recall, mean_f1, best_score_threshold = -1.0, -1.0, -1.0, -1.0, -1.0
        accumulated_matching_info = getattr(self, self.state_key)
        best_score_threshold_per_cls = np.zeros(self.num_cls)
        mean_ap_per_class = np.zeros(self.num_cls)

        if len(accumulated_matching_info):
            matching_info_tensors = [torch.cat(x, 0) for x in list(zip(*accumulated_matching_info))]

            # shape (n_class, nb_iou_thresh)
            (
                ap_per_present_classes,
                precision_per_present_classes,
                recall_per_present_classes,
                f1_per_present_classes,
                present_classes,
                best_score_threshold,
                best_score_thresholds_per_present_classes,
            ) = compute_detection_metrics(
                *matching_info_tensors,
                recall_thresholds=self.recall_thresholds,
                score_threshold=self.score_threshold,
                device="cpu" if self.accumulate_on_cpu else self.device,
            )

            # Precision, recall and f1 are computed for IoU threshold range, averaged over classes
            # results before version 3.0.4 (Dec 11 2022) were computed only for smallest value (i.e IoU 0.5 if metric is @0.5:0.95)
            mean_precision, mean_recall, mean_f1 = precision_per_present_classes.mean(), recall_per_present_classes.mean(), f1_per_present_classes.mean()

            # MaP is averaged over IoU thresholds and over classes
            mean_ap = ap_per_present_classes.mean()

            # Fill array of per-class AP scores with values for classes that were present in the dataset
            ap_per_class = ap_per_present_classes.mean(1)
            for i, class_index in enumerate(present_classes):
                mean_ap_per_class[class_index] = float(ap_per_class[i])
                best_score_threshold_per_cls[class_index] = float(best_score_thresholds_per_present_classes[i])

        output_dict = {
            self.precision_metric_key: float(mean_precision),
            self.recall_metric_key: float(mean_recall),
            self.map_metric_key: float(mean_ap),
            self.f1_metric_key: float(mean_f1),
        }

        if self.include_classwise_ap:
            for i, ap_i in enumerate(mean_ap_per_class):
                output_dict[self.per_class_ap_names[i]] = float(ap_i)

        if self.calc_best_score_thresholds:
            output_dict["Best_score_threshold"] = float(best_score_threshold)

        if self.include_classwise_ap and self.calc_best_score_thresholds:
            for threshold_per_class_names, threshold_value in zip(self.best_threshold_per_class_names, best_score_threshold_per_cls):
                output_dict[threshold_per_class_names] = float(threshold_value)

        return output_dict

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        """
        When in distributed mode, stats are aggregated after each forward pass to the metric state. Since these have all
        different sizes we override the synchronization function since it works only for tensors (and use
        all_gather_object)
        :param dist_sync_fn:
        :return:
        """
        if self.world_size is None:
            self.world_size = super_gradients.common.environment.ddp_utils.get_world_size() if self.is_distributed else -1
        if self.rank is None:
            self.rank = torch.distributed.get_rank() if self.is_distributed else -1

        if self.is_distributed:
            local_state_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
            gathered_state_dicts = [None] * self.world_size
            torch.distributed.barrier()
            torch.distributed.all_gather_object(gathered_state_dicts, local_state_dict)
            matching_info = []
            for state_dict in gathered_state_dicts:
                matching_info += state_dict[self.state_key]
            matching_info = tensor_container_to_device(matching_info, device="cpu" if self.accumulate_on_cpu else self.device)

            setattr(self, self.state_key, matching_info)

    def _get_range_str(self):
        return "@%.2f" % self.iou_thresholds[0] if not len(self.iou_thresholds) > 1 else "@%.2f:%.2f" % (self.iou_thresholds[0], self.iou_thresholds[-1])


@register_metric(Metrics.DETECTION_METRICS_DISTANCE_BASED)
class DetectionMetricsDistanceBased(DetectionMetrics):
    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback,
        normalize_targets: bool = False,
        distance_thresholds: List[float] = [5.0],
        distance_metric: DistanceMetric = EuclideanDistance(),
        recall_thres: torch.Tensor = None,
        score_thres: float = 0.1,
        top_k_predictions: int = 100,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        calc_best_score_thresholds: bool = True,
        include_classwise_ap: bool = False,
        class_names: List[str] = None,
    ):
        self.distance_thresholds = distance_thresholds
        self.distance_metric = distance_metric
        super().__init__(
            num_cls=num_cls,
            post_prediction_callback=post_prediction_callback,
            normalize_targets=normalize_targets,
            recall_thres=recall_thres,
            score_thres=score_thres,
            top_k_predictions=top_k_predictions,
            dist_sync_on_step=dist_sync_on_step,
            accumulate_on_cpu=accumulate_on_cpu,
            calc_best_score_thresholds=calc_best_score_thresholds,
            include_classwise_ap=include_classwise_ap,
            class_names=class_names,
            state_dict_prefix="distance_based_",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor, device: str, inputs: torch.tensor, crowd_targets: Optional[torch.Tensor] = None) -> None:
        """
        Apply NMS and match all the predictions and targets of a given batch, and update the metric state accordingly.
        Use distance-based definition of true positives.

        :param preds: torch.Tensor: Raw output of the model. The format might change from one model to another,
                                    but has to fit the input format of the post_prediction_callback (cx, cy, wh).
        :param target: torch.Tensor: Targets for all images of shape (total_num_targets, 6) LABEL_CXCYWH.
                                      Format:  (index, label, cx, cy, w, h)
        :param device: str: Device to run on.
        :param inputs: torch.Tensor: Input image tensor of shape (batch_size, n_img, height, width).
        :param crowd_targets: Optional[torch.Tensor]: Crowd targets for all images of shape (total_num_targets, 6), LABEL_CXCYWH.
        """
        _, _, height, width = inputs.shape

        distance_matcher = DistanceMatching(self.distance_metric, self.distance_thresholds)

        targets = target.clone()
        crowd_targets = torch.zeros(size=(0, 6), device=device) if crowd_targets is None else crowd_targets.clone()

        preds = self.post_prediction_callback(preds, device=device)

        new_matching_info = compute_detection_matching(
            output=preds,
            targets=targets,
            height=height,
            width=width,
            crowd_targets=crowd_targets,
            top_k=self.top_k_predictions,
            denormalize_targets=self.denormalize_targets,
            device=self.device,
            return_on_cpu=self.accumulate_on_cpu,
            matching_strategy=distance_matcher,
        )

        accumulated_matching_info = getattr(self, self.state_key)
        setattr(self, self.state_key, accumulated_matching_info + new_matching_info)

    def _get_range_str(self):
        return (
            "@DIST%.2f" % self.distance_thresholds[0]
            if not len(self.distance_thresholds) > 1
            else "@DIST%.2f:%.2f" % (self.distance_thresholds[0], self.distance_thresholds[-1])
        )


@register_metric(Metrics.DETECTION_METRICS_050)
class DetectionMetrics_050(DetectionMetrics):
    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback = None,
        normalize_targets: bool = False,
        recall_thres: torch.Tensor = None,
        score_thres: float = 0.1,
        top_k_predictions: int = 100,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        calc_best_score_thresholds: bool = True,
        include_classwise_ap: bool = False,
        class_names: List[str] = None,
    ):
        super().__init__(
            num_cls=num_cls,
            post_prediction_callback=post_prediction_callback,
            normalize_targets=normalize_targets,
            iou_thres=IouThreshold.MAP_05,
            recall_thres=recall_thres,
            score_thres=score_thres,
            top_k_predictions=top_k_predictions,
            dist_sync_on_step=dist_sync_on_step,
            accumulate_on_cpu=accumulate_on_cpu,
            calc_best_score_thresholds=calc_best_score_thresholds,
            include_classwise_ap=include_classwise_ap,
            class_names=class_names,
        )


@register_metric(Metrics.DETECTION_METRICS_075)
class DetectionMetrics_075(DetectionMetrics):
    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback = None,
        normalize_targets: bool = False,
        recall_thres: torch.Tensor = None,
        score_thres: float = 0.1,
        top_k_predictions: int = 100,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        calc_best_score_thresholds: bool = True,
        include_classwise_ap: bool = False,
        class_names: List[str] = None,
    ):
        super().__init__(
            num_cls=num_cls,
            post_prediction_callback=post_prediction_callback,
            normalize_targets=normalize_targets,
            iou_thres=0.75,
            recall_thres=recall_thres,
            score_thres=score_thres,
            top_k_predictions=top_k_predictions,
            dist_sync_on_step=dist_sync_on_step,
            accumulate_on_cpu=accumulate_on_cpu,
            calc_best_score_thresholds=calc_best_score_thresholds,
            include_classwise_ap=include_classwise_ap,
            class_names=class_names,
        )


@register_metric(Metrics.DETECTION_METRICS_050_095)
class DetectionMetrics_050_095(DetectionMetrics):
    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback = None,
        normalize_targets: bool = False,
        recall_thres: torch.Tensor = None,
        score_thres: float = 0.1,
        top_k_predictions: int = 100,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        calc_best_score_thresholds: bool = True,
        include_classwise_ap: bool = False,
        class_names: List[str] = None,
    ):
        super().__init__(
            num_cls=num_cls,
            post_prediction_callback=post_prediction_callback,
            normalize_targets=normalize_targets,
            iou_thres=IouThreshold.MAP_05_TO_095,
            recall_thres=recall_thres,
            score_thres=score_thres,
            top_k_predictions=top_k_predictions,
            dist_sync_on_step=dist_sync_on_step,
            accumulate_on_cpu=accumulate_on_cpu,
            calc_best_score_thresholds=calc_best_score_thresholds,
            include_classwise_ap=include_classwise_ap,
            class_names=class_names,
        )
