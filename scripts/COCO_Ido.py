from typing import Dict, Union
from torchmetrics import Metric
import super_gradients
from super_gradients.common.registry import register_metric
from super_gradients.training.utils import tensor_container_to_device
from super_gradients.training.utils.detection_utils import compute_detection_matching
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback, IouThreshold
from super_gradients.common.abstractions.abstract_logger import get_logger
import numpy as np
from typing import Tuple, Optional
import torch


logger = get_logger(__name__)


def compute_detection_metrics(
    preds_matched: torch.Tensor,
    preds_to_ignore: torch.Tensor,
    preds_scores: torch.Tensor,
    preds_cls: torch.Tensor,
    targets_cls: torch.Tensor,
    device: str,
    recall_thresholds: Optional[torch.Tensor] = None,
    score_threshold: Optional[float] = 0.1,
) -> Tuple:
    """
    Compute the list of precision, recall, MaP and f1 for every recall IoU threshold and for every class.

    :param preds_matched:      Tensor of shape (num_predictions, n_iou_thresholds)
                                    True when prediction (i) is matched with a target with respect to the (j)th IoU threshold
    :param preds_to_ignore     Tensor of shape (num_predictions, n_iou_thresholds)
                                    True when prediction (i) is matched with a crowd target with respect to the (j)th IoU threshold
    :param preds_scores:       Tensor of shape (num_predictions), confidence score for every prediction
    :param preds_cls:          Tensor of shape (num_predictions), predicted class for every prediction
    :param targets_cls:        Tensor of shape (num_targets), ground truth class for every target box to be detected
    :param recall_thresholds:   Recall thresholds used to compute MaP.
    :param score_threshold:    Minimum confidence score to consider a prediction for the computation of
                                    precision, recall and f1 (not MaP)
    :param device:             Device

    :return:
        :ap, precision, recall, f1: Tensors of shape (n_class, nb_iou_thrs)
        :unique_classes:            Vector with all unique target classes
    """
    preds_matched, preds_to_ignore = preds_matched.to(device), preds_to_ignore.to(device)
    preds_scores, preds_cls, targets_cls = preds_scores.to(device), preds_cls.to(device), targets_cls.to(device)

    recall_thresholds = torch.linspace(0, 1, 101, device=device) if recall_thresholds is None else recall_thresholds.to(device)

    unique_classes = torch.unique(targets_cls)
    n_class, nb_iou_thrs = len(unique_classes), preds_matched.shape[-1]

    ap = torch.zeros((n_class, nb_iou_thrs), device=device)
    precision = torch.zeros((n_class, nb_iou_thrs), device=device)
    recall = torch.zeros((n_class, nb_iou_thrs), device=device)
    all_precision = torch.zeros((n_class, len(recall_thresholds), nb_iou_thrs), device=device)

    for cls_i, cls in enumerate(unique_classes):
        cls_preds_idx, cls_targets_idx = (preds_cls == cls), (targets_cls == cls)
        cls_all_precision, cls_ap, cls_precision, cls_recall = compute_detection_metrics_per_cls(
            preds_matched=preds_matched[cls_preds_idx],
            preds_to_ignore=preds_to_ignore[cls_preds_idx],
            preds_scores=preds_scores[cls_preds_idx],
            n_targets=cls_targets_idx.sum(),
            recall_thresholds=recall_thresholds,
            score_threshold=score_threshold,
            device=device,
        )
        ap[cls_i, :] = cls_ap
        precision[cls_i, :] = cls_precision
        recall[cls_i, :] = cls_recall
        all_precision[cls_i, :, :] = cls_all_precision

    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return all_precision, ap, precision, recall, f1, unique_classes


def compute_detection_metrics_per_cls(
    preds_matched: torch.Tensor,
    preds_to_ignore: torch.Tensor,
    preds_scores: torch.Tensor,
    n_targets: int,
    recall_thresholds: torch.Tensor,
    score_threshold: float,
    device: str,
):
    """
    Compute the list of precision, recall and MaP of a given class for every recall IoU threshold.

        :param preds_matched:      Tensor of shape (num_predictions, n_iou_thresholds)
                                        True when prediction (i) is matched with a target
                                        with respect to the(j)th IoU threshold
        :param preds_to_ignore     Tensor of shape (num_predictions, n_iou_thresholds)
                                        True when prediction (i) is matched with a crowd target
                                        with respect to the (j)th IoU threshold
        :param preds_scores:       Tensor of shape (num_predictions), confidence score for every prediction
        :param n_targets:          Number of target boxes of this class
        :param recall_thresholds:  Tensor of shape (max_n_rec_thresh) list of recall thresholds used to compute MaP
        :param score_threshold:    Minimum confidence score to consider a prediction for the computation of
                                        precision and recall (not MaP)
        :param device:             Device

        :return ap, precision, recall:  Tensors of shape (nb_iou_thrs)
    """
    nb_iou_thrs = preds_matched.shape[-1]

    tps = preds_matched
    fps = torch.logical_and(torch.logical_not(preds_matched), torch.logical_not(preds_to_ignore))

    if len(tps) == 0:
        return torch.zeros((len(recall_thresholds), nb_iou_thrs), device=device), torch.zeros(nb_iou_thrs, device=device), 0, 0

    # Sort by decreasing score
    dtype = torch.uint8 if preds_scores.is_cuda and preds_scores.dtype is torch.bool else preds_scores.dtype
    sort_ind = torch.argsort(preds_scores.to(dtype), descending=True)
    tps = tps[sort_ind, :]
    fps = fps[sort_ind, :]
    preds_scores = preds_scores[sort_ind].contiguous()

    # Rolling sum over the predictions
    rolling_tps = torch.cumsum(tps, axis=0, dtype=torch.float)
    rolling_fps = torch.cumsum(fps, axis=0, dtype=torch.float)

    rolling_recalls = rolling_tps / n_targets
    rolling_precisions = rolling_tps / (rolling_tps + rolling_fps + torch.finfo(torch.float64).eps)

    # Reversed cummax to only have decreasing values
    rolling_precisions = rolling_precisions.flip(0).cummax(0).values.flip(0)

    # ==================
    # RECALL & PRECISION

    # We want the rolling precision/recall at index i so that: preds_scores[i-1] >= score_threshold > preds_scores[i]
    # Note: torch.searchsorted works on increasing sequence and preds_scores is decreasing, so we work with "-"
    lowest_score_above_threshold = torch.searchsorted(-preds_scores, -score_threshold, right=False)

    if lowest_score_above_threshold == 0:  # Here score_threshold > preds_scores[0], so no pred is above the threshold
        recall = 0
        precision = 0  # the precision is not really defined when no pred but we need to give it a value
    else:
        recall = rolling_recalls[lowest_score_above_threshold - 1]
        precision = rolling_precisions[lowest_score_above_threshold - 1]

    # ==================
    # AVERAGE PRECISION

    # shape = (nb_iou_thrs, n_recall_thresholds)
    recall_thresholds = recall_thresholds.view(1, -1).repeat(nb_iou_thrs, 1)

    # We want the index i so that: rolling_recalls[i-1] < recall_thresholds[k] <= rolling_recalls[i]
    # Note:  when recall_thresholds[k] > max(rolling_recalls), i = len(rolling_recalls)
    # Note2: we work with transpose (.T) to apply torch.searchsorted on first dim instead of the last one
    recall_threshold_idx = torch.searchsorted(rolling_recalls.T.contiguous(), recall_thresholds, right=False).T

    # When recall_thresholds[k] > max(rolling_recalls), rolling_precisions[i] is not defined, and we want precision = 0
    rolling_precisions = torch.cat((rolling_precisions, torch.zeros(1, nb_iou_thrs, device=device)), dim=0)

    # shape = (n_recall_thresholds, nb_iou_thrs)
    sampled_precision_points = torch.gather(input=rolling_precisions, index=recall_threshold_idx, dim=0)

    # Average over the recall_thresholds
    ap = sampled_precision_points.mean(0)

    return sampled_precision_points, ap, precision, recall


logger = get_logger(__name__)


@register_metric("my_detection_metric")
class DetectionMetrics(Metric):
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
        accumulate_on_cpu:     Run on CPU regardless of device used in other parts.
                            This is to avoid "CUDA out of memory" that might happen on GPU (default False)
    """

    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback,
        normalize_targets: bool = False,
        iou_thres: Union[IouThreshold, float] = IouThreshold.MAP_05_TO_095,
        recall_thres: torch.Tensor = None,
        score_thres: float = 0.1,
        top_k_predictions: int = 100,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        use_cocoapi: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_cls = num_cls
        self.iou_thres = iou_thres

        if isinstance(iou_thres, IouThreshold):
            self.iou_thresholds = iou_thres.to_tensor()
        else:
            self.iou_thresholds = torch.tensor([iou_thres])

        self.map_str = "mAP" + self._get_range_str()
        self.greater_component_is_better = {
            # f"Precision{self._get_range_str()}": True,
            # f"Recall{self._get_range_str()}": True,
            f"mAP{self._get_range_str()}": True,
            # f"F1{self._get_range_str()}": True,
            f"mAP@{self.iou_thresholds[0]:.2f}": True,
            f"mAP@{self.iou_thresholds[5]:.2f}": True,
        }
        self.component_names = list(self.greater_component_is_better.keys())
        self.components = len(self.component_names)

        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()
        self.denormalize_targets = not normalize_targets
        self.world_size = None
        self.rank = None
        self.add_state(f"matching_info{self._get_range_str()}", default=[], dist_reduce_fx=None)

        self.recall_thresholds = torch.linspace(0, 1, 101) if recall_thres is None else recall_thres
        self.score_threshold = score_thres
        self.top_k_predictions = top_k_predictions

        self.accumulate_on_cpu = accumulate_on_cpu
        self.pred_results = []
        self.img_ids = []
        self.use_cocoapi = use_cocoapi

    def update(self, preds, target: torch.Tensor, device: str, inputs: torch.tensor, paths, shapes, crowd_targets: Optional[torch.Tensor] = None):
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
            preds,
            targets,
            height,
            width,
            self.iou_thresholds,
            crowd_targets=crowd_targets,
            top_k=self.top_k_predictions,
            denormalize_targets=self.denormalize_targets,
            device=self.device,
            return_on_cpu=self.accumulate_on_cpu,
        )

        accumulated_matching_info = getattr(self, f"matching_info{self._get_range_str()}")
        setattr(self, f"matching_info{self._get_range_str()}", accumulated_matching_info + new_matching_info)
        if self.use_cocoapi:
            ids = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                27,
                28,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                67,
                70,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
            ]
            self.pred_results += self.convert_to_coco_format(preds, inputs, paths, shapes, ids)  # noqa  # noqa

            img_ids = [int(os.path.basename(x).split(".")[0]) for x in paths]
            self.img_ids += img_ids

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """Rescale coords (xyxy) from img1_shape to img0_shape."""
        if ratio_pad is None:  # calculate from img0_shape
            gain = [min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])]  # gain  = old / new
            # if self.scale_exact:
            if False:
                gain = [img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]]
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        # if self.scale_exact:
        if False:
            coords[:, [0, 2]] /= gain[1]  # x gain
        else:
            coords[:, [0, 2]] /= gain[0]  # raw x gain
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, [1, 3]] /= gain[0]  # y gain

        if isinstance(coords, torch.Tensor):  # faster individually
            coords[:, 0].clamp_(0, img0_shape[1])  # x1
            coords[:, 1].clamp_(0, img0_shape[0])  # y1
            coords[:, 2].clamp_(0, img0_shape[1])  # x2
            coords[:, 3].clamp_(0, img0_shape[0])  # y2
        else:  # np.array (faster grouped)
            coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
            coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
        return coords

    def box_convert(self, x):
        """Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right."""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def convert_to_coco_format(self, outputs, imgs, paths, shapes, ids):
        pred_results = []
        for i, pred in enumerate(outputs):
            if pred is None or len(pred) == 0:
                continue
            from pathlib import Path

            path, shape = Path(paths[i]), shapes[i][0]  # noqa
            ratio_pad = shapes[i][1]  # noqa
            self.scale_coords(imgs[i].shape[1:], pred[:, :4], shape, shapes[i][1])
            # image_id = int(path.stem) if self.is_coco else path.stem
            image_id = int(path.stem)
            bboxes = self.box_convert(pred[:, 0:4])
            bboxes[:, :2] -= bboxes[:, 2:] / 2
            cls = pred[:, 5]
            scores = pred[:, 4]
            for ind in range(pred.shape[0]):
                category_id = ids[int(cls[ind])]
                bbox = [round(x, 3) for x in bboxes[ind].tolist()]
                score = round(scores[ind].item(), 5)
                pred_data = {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}
                pred_results.append(pred_data)
        return pred_results

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute the metrics for all the accumulated results.
        :return: Metrics of interest
        """
        mean_ap, mean_precision, mean_recall, mean_f1 = 0.0, 0.0, 0.0, 0.0  # noqa
        all_precision = None
        accumulated_matching_info = getattr(self, f"matching_info{self._get_range_str()}")

        # if len(accumulated_matching_info):
        if len(accumulated_matching_info[0]) and (self.rank is None or self.rank <= 0):
            matching_info_tensors = (
                accumulated_matching_info if self.rank is not None and self.rank == 0 else [torch.cat(x, 0) for x in list(zip(*accumulated_matching_info))]
            )
            # matching_info_tensors = [torch.cat(x, 0) for x in list(zip(*accumulated_matching_info))]

            # shape (n_class, nb_iou_thresh)
            all_precision, ap, precision, recall, f1, unique_classes = compute_detection_metrics(
                *matching_info_tensors,
                recall_thresholds=self.recall_thresholds,
                score_threshold=self.score_threshold,
                device="cpu" if self.accumulate_on_cpu else self.device,
            )

            # Precision, recall and f1 are computed for IoU threshold range, averaged over classes
            # results before version 3.0.4 (Dec 11 2022) were computed only for smallest value (i.e IoU 0.5 if metric is @0.5:0.95)
            # mean_precision, mean_recall, mean_f1 = precision.mean(), recall.mean(), f1.mean()

            # MaP is averaged over IoU thresholds and over classes
            mean_ap = ap.mean()

        if self.use_cocoapi:
            pred_json = "/home/ido.shahaf/predictions.json"
            anno_json = "/data/coco/annotations/instances_val2017.json"
            import ujson as json
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            with open(pred_json, "w") as f:
                json.dump(self.pred_results, f)
            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            cocoEval = COCOeval(anno, pred, "bbox")
            imgIds = self.img_ids
            cocoEval.params.imgIds = imgIds
            cocoEval.params.areaRng = [cocoEval.params.areaRng[0]]
            cocoEval.params.maxDets = [100]
            cocoEval.evaluate()
            cocoEval.accumulate()
            s = cocoEval.eval["precision"]
            s50 = s[0]
            map, map50 = s[s > -1].mean(), s50[s50 > -1].mean()

            print(f"mAP:   COCO={map}, SG={mean_ap.item()}")
            print(f"mAP50: COCO={map50}, SG={all_precision[..., 0].mean().item()}")
            print(mean_ap.item())

        world_size = 1 if self.world_size is None else self.world_size

        return {
            # f"Precision{self._get_range_str()}": mean_precision,
            # f"Recall{self._get_range_str()}": mean_recall,
            f"mAP{self._get_range_str()}": mean_ap * world_size,
            f"mAP@{self.iou_thresholds[0]:.2f}": 0.0 if all_precision is None else all_precision[..., 0].mean() * world_size,
            f"mAP@{self.iou_thresholds[5]:.2f}": 0.0 if all_precision is None else all_precision[..., 5].mean() * world_size,
            # f"F1{self._get_range_str()}": mean_f1,
        }

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
            matching_info_attr_name = f"matching_info{self._get_range_str()}"

            bla = [torch.cat(x, 0) for x in zip(*getattr(self, matching_info_attr_name))]
            gathered_blas = [None] * self.world_size
            torch.distributed.barrier()
            torch.distributed.all_gather_object(gathered_blas, bla)
            if self.rank == 0:
                gathered_blas2 = tensor_container_to_device(gathered_blas, device="cpu" if self.accumulate_on_cpu else self.device)
                gathered_blas3 = [torch.cat(x, 0) for x in zip(*gathered_blas2)]

                setattr(self, matching_info_attr_name, gathered_blas3)

        # if self.is_distributed:
        #     local_state_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
        #     gathered_state_dicts = [None] * self.world_size
        #     torch.distributed.barrier()
        #     torch.distributed.all_gather_object(gathered_state_dicts, local_state_dict)
        #     matching_info = []
        #     for state_dict in gathered_state_dicts:
        #         matching_info += state_dict[f"matching_info{self._get_range_str()}"]
        #     matching_info = tensor_container_to_device(matching_info, device="cpu" if self.accumulate_on_cpu else self.device)
        #
        #     setattr(self, f"matching_info{self._get_range_str()}", matching_info)

    def _get_range_str(self):
        return "@%.2f" % self.iou_thresholds[0] if not len(self.iou_thresholds) > 1 else "@%.2f:%.2f" % (self.iou_thresholds[0], self.iou_thresholds[-1])


if __name__ == "__main__":

    from super_gradients.training.dataloaders.dataloaders import (
        coco2017_val_yolo_nas,
    )
    from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

    post_prediction_callback = PPYoloEPostPredictionCallback(
        score_threshold=0.01,
        nms_top_k=1000,
        max_predictions=300,
        nms_threshold=0.7,
    )
    import os  # noqa
    from super_gradients.training import models, Trainer  # noqa
    from super_gradients.common.object_names import Models  # noqa

    dataloader = coco2017_val_yolo_nas(
        dataloader_params={"collate_fn": CrowdDetectionCollateFN(), "batch_size": 10},
        dataset_params={"with_crowd": True, "ignore_empty_annotations": False},
    )

    trainer = Trainer(Models.PP_YOLOE_S)
    model = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
    # model.eval()

    res = trainer.test(
        model=model,
        test_loader=dataloader,
        test_metrics_list=[DetectionMetrics(post_prediction_callback=post_prediction_callback, num_cls=80, normalize_targets=True, use_cocoapi=True)],
    )

    print(res)
