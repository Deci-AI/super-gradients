# This file contains modified version of COCOEval class from pycocotools library,
# which is used to compute mAP metric for pose estimation task.

import numpy as np
import dataclasses

from collections import defaultdict
from typing import Union, List, Any, Tuple, Mapping

import torch
from pycocotools.coco import COCO

from super_gradients.training.utils.detection_utils import compute_detection_metrics_per_cls


def compute_visible_bbox_xywh(joints: np.ndarray, visibility_mask: np.ndarray) -> np.ndarray:
    """
    Compute the bounding box (X,Y,W,H) of the visible joints for each instance.

    :param joints:  [Num Instances, Num Joints, 2+] last channel must have dimension of
                    at least 2 that is considered to contain (X,Y) coordinates of the keypoint
    :param visibility_mask: [Num Instances, Num Joints]
    :return: A numpy array [Num Instances, 4] where last dimension contains bbox in format XYWH
    """
    visibility_mask = visibility_mask > 0
    initial_value = 1_000_000

    x1 = np.min(joints[:, :, 0], where=visibility_mask, initial=initial_value, axis=-1)
    y1 = np.min(joints[:, :, 1], where=visibility_mask, initial=initial_value, axis=-1)

    x1[x1 == initial_value] = 0
    y1[y1 == initial_value] = 0

    x2 = np.max(joints[:, :, 0], where=visibility_mask, initial=0, axis=-1)
    y2 = np.max(joints[:, :, 1], where=visibility_mask, initial=0, axis=-1)

    w = x2 - x1
    h = y2 - y1

    return np.stack([x1, y1, w, h], axis=-1)


@dataclasses.dataclass
class EvaluationParams:
    """
    Params for computing pose estimation metrics
    """

    iou_thresholds: np.ndarray
    recall_thresholds: np.ndarray
    maxDets: int
    useCats: bool
    sigmas: np.ndarray

    @classmethod
    def get_predefined_coco_params(cls):
        """
        Create evaluation params for COCO dataset
        :return:
        """
        return cls(
            iou_thresholds=np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True),
            recall_thresholds=np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True),
            maxDets=20,
            useCats=True,
            sigmas=np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0,
        )


@dataclasses.dataclass
class ImageLevelEvaluationResult:
    image_id: Union[str, int]
    category_id: int
    dtMatches: Any
    gtMatches: Any

    dtScores: List
    dtIgnore: Any

    gtIgnore: Any
    gtIsCrowd: np.ndarray


@dataclasses.dataclass
class DatasetLevelEvaluationResult:
    params: EvaluationParams
    counts: Tuple[int, int, int]
    precision: np.ndarray
    recall: np.ndarray

    @property
    def ap_metric(self):
        return self._summarize(1)

    @property
    def ar_metric(self):
        return self._summarize(1)

    def all_metrics(self):
        return {
            "AP": self._summarize(1),
            "AP_0.5": self._summarize(1, iouThr=0.5),
            "AP_0.75": self._summarize(1, iouThr=0.75),
            "AR": self._summarize(0),
            "AR_0.5": self._summarize(0, iouThr=0.5),
            "AR_0.75": self._summarize(0, iouThr=0.75),
        }

    def print(self):
        p = self.params

        def _print_summarize(ap=1, iouThr=None):
            score = self._summarize(ap, iouThr)
            iStr = " {:<18} {} @[ IoU={:<9} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = "{:0.2f}:{:0.2f}".format(p.iou_thresholds[0], p.iou_thresholds[-1]) if iouThr is None else "{:0.2f}".format(iouThr)
            print(iStr.format(titleStr, typeStr, iouStr, score))

        _print_summarize(1)
        _print_summarize(1, iouThr=0.5)
        _print_summarize(1, iouThr=0.75)
        _print_summarize(0)
        _print_summarize(0, iouThr=0.5)
        _print_summarize(0, iouThr=0.75)

    def _summarize(self, ap=1, iouThr=None):
        p = self.params

        if ap == 1:
            s = self.precision
        else:
            s = self.recall

        if iouThr is not None:
            t = np.where(iouThr == p.iou_thresholds)[0]
            s = s[t]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        return mean_s


def computeKeypointsIoU(
    pred_joints: Union[np.ndarray, List[np.ndarray]],
    pred_scores: Union[np.ndarray, List[float]],
    gt_joints: Union[np.ndarray, List[np.ndarray]],
    gt_keypoint_visibility: Union[np.ndarray, List[float]],
    sigmas: np.ndarray,
    max_dets: int = 20,
    gt_areas: np.ndarray = None,
    gt_bboxes: np.ndarray = None,
) -> np.ndarray:
    """

    :param pred_joints: [K, NumJoints, 2] or [K, NumJoints, 3]
    :param pred_scores: [K]
    :param gt_joints:   [M, NumJoints, 2]
    :param gt_keypoint_visibility: [M, NumJoints]
    :param gt_areas: [M] Area of each ground truth instance. COCOEval uses area of the instance mask to scale OKs, so it must be provided separately.
        If None, we will use area of bounding box of each instance computed from gt_joints.

    :param gt_bboxes: [M, 4] Bounding box (X,Y,W,H) of each ground truth instance. If None, we will use bounding box of each instance computed from gt_joints.
    :param sigmas: [NumJoints]
    :return: IoU matrix [min(K, max_dets), M]
    """

    # Sort predictions by score from highest to lowest and retain only the top max_dets
    pred_scores = np.array(pred_scores)
    inds = np.argsort(-pred_scores, kind="mergesort")
    pred_joints = [pred_joints[i] for i in inds]

    if len(pred_joints) > max_dets:
        pred_joints = pred_joints[0:max_dets]

    if len(gt_joints) == 0 or len(pred_joints) == 0:
        return np.zeros((len(pred_joints), len(gt_joints)))

    ious = np.zeros((len(pred_joints), len(gt_joints)))
    vars = (sigmas * 2) ** 2
    num_joints = len(sigmas)

    if gt_bboxes is None:
        gt_bboxes = compute_visible_bbox_xywh(gt_joints, gt_keypoint_visibility)

    if gt_areas is None:
        gt_areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]

    # compute oks between each detection and ground truth object
    for gt_index, (gt_keypoints, gt_keypoint_visibility, gt_bbox, gt_area) in enumerate(zip(gt_joints, gt_keypoint_visibility, gt_bboxes, gt_areas)):
        # create bounds for ignore regions(double the gt bbox)
        xg = gt_keypoints[:, 0]
        yg = gt_keypoints[:, 1]
        k1 = np.count_nonzero(gt_keypoint_visibility > 0)

        x0 = gt_bbox[0] - gt_bbox[2]
        x1 = gt_bbox[0] + gt_bbox[2] * 2
        y0 = gt_bbox[1] - gt_bbox[3]
        y1 = gt_bbox[1] + gt_bbox[3] * 2

        for pred_index, pred_keypoints in enumerate(pred_joints):
            xd = pred_keypoints[:, 0]
            yd = pred_keypoints[:, 1]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((num_joints))
                dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)

            e = (dx**2 + dy**2) / vars / (gt_area + np.spacing(1)) / 2

            if k1 > 0:
                e = e[gt_keypoint_visibility > 0]
            ious[pred_index, gt_index] = np.sum(np.exp(-e)) / e.shape[0]

    return ious


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, params: EvaluationParams):
        self.params = params

    def evaluate_from_coco(self, groundtruth: COCO, predictions: COCO):
        """

        :param groundtruth: COCO-like object with ground truth annotations
        :param predictions: COCO-like object with predictions
        :return:
        """
        imgIds = list(sorted(groundtruth.getImgIds()))
        catIds = list(np.unique(groundtruth.getCatIds()))

        if self.params.useCats:
            gts = groundtruth.loadAnns(groundtruth.getAnnIds(imgIds=imgIds, catIds=catIds))
            dts = predictions.loadAnns(predictions.getAnnIds(imgIds=imgIds, catIds=catIds))
        else:
            gts = groundtruth.loadAnns(groundtruth.getAnnIds(imgIds=imgIds))
            dts = predictions.loadAnns(predictions.getAnnIds(imgIds=imgIds))

        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            gt["ignore"] = (gt["num_keypoints"] == 0) or bool(gt["ignore"])

        _gts = defaultdict(list)  # gt for evaluation
        _dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            _gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            _dts[dt["image_id"], dt["category_id"]].append(dt)

        catIds = catIds if self.params.useCats else [-1]

        # ious between all gts and dts
        ious = {(imgId, catId): self.computeOksV2(_gts[imgId, catId], _dts[imgId, catId]) for imgId in imgIds for catId in catIds}

        evalImgs = [self.evaluateImg(imgId, catId, self.params.maxDets, catIds, ious, _gts, _dts) for catId in catIds for imgId in imgIds]
        # result = self.accumulate_with_coco(evalImgs, imgIds, catIds)

        return evalImgs, imgIds, catIds

    def computeOksV2(self, groundtruths, predictions):

        # TODO: Move this part to evaluate_from_coco
        pred_keypoints = np.array([np.array(dt["keypoints"]).reshape(-1, 3) for dt in predictions])
        pred_scores = np.array([dt["score"] for dt in predictions])

        gt_keypoints_with_visibility = np.array([np.array(gt["keypoints"]).reshape(-1, 3) for gt in groundtruths])
        gt_joints = gt_keypoints_with_visibility[:, :, 0:2] if len(gt_keypoints_with_visibility) else []
        gt_keypoint_visibility = gt_keypoints_with_visibility[:, :, 2] if len(gt_keypoints_with_visibility) else []
        gt_areas = np.array([gt["area"] for gt in groundtruths])
        gt_bboxes = np.array([gt["bbox"] for gt in groundtruths])
        # TODO END: Move this part to evaluate_from_coco

        ious = computeKeypointsIoU(
            pred_joints=pred_keypoints,
            pred_scores=pred_scores,
            gt_joints=gt_joints,
            gt_keypoint_visibility=gt_keypoint_visibility,
            gt_areas=gt_areas,
            gt_bboxes=gt_bboxes,
            max_dets=self.params.maxDets,
            sigmas=self.params.sigmas,
        )
        return ious

    def evaluateImg(
        self, imgId: int, catId: int, maxDet, catIds, ious, _gts: Mapping[Tuple[int, int], Any], _dts: Mapping[Tuple[int, int], Any]
    ) -> ImageLevelEvaluationResult:
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = _gts[imgId, catId]
            dt = _dts[imgId, catId]
        else:
            gt = [_ for cId in catIds for _ in _gts[imgId, cId]]
            dt = [_ for cId in catIds for _ in _dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = np.array([bool(o["iscrowd"]) for o in gt])
        # load computed ious
        ious = ious[imgId, catId][:, gtind] if len(ious[imgId, catId]) > 0 else ious[imgId, catId]

        T = len(p.iou_thresholds)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iou_thresholds):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

        # store results for given image and category
        return ImageLevelEvaluationResult(
            image_id=imgId,
            category_id=catId,
            dtMatches=dtm,
            gtMatches=gtm,
            dtScores=[d["score"] for d in dt],
            gtIgnore=gtIg,
            gtIsCrowd=iscrowd,
            dtIgnore=dtIg,
        )

    def accumulate_with_coco(self, evalImgs, imgIds, catIds):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """

        p = self.params

        T = len(p.iou_thresholds)
        R = len(p.recall_thresholds)
        K = len(catIds)

        precision = -np.ones((T, R, K))  # -1 for the precision of absent categories
        recall = -np.ones((T, K))

        # create dictionary for future indexing
        setK = set(catIds)
        setI = set(imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(catIds) if k in setK]
        i_list = [n for n, i in enumerate(imgIds) if i in setI]
        I0 = len(imgIds)

        # retrieve E at each category
        for k, k0 in enumerate(k_list):
            Nk = k0 * I0
            E = [evalImgs[Nk + i] for i in i_list]
            E: List[ImageLevelEvaluationResult] = [e for e in E if e is not None]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e.dtScores[0 : p.maxDets] for e in E])

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dtScores, kind="mergesort")

            dtm = np.concatenate([e.dtMatches[:, 0 : p.maxDets] for e in E], axis=1)[:, inds]
            dtIg = np.concatenate([e.dtIgnore[:, 0 : p.maxDets] for e in E], axis=1)[:, inds]
            gtIg = np.concatenate([e.gtIgnore for e in E])
            npig = np.count_nonzero(gtIg == 0)
            if npig == 0:
                continue
            tps = np.logical_and(dtm, np.logical_not(dtIg))
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                q = np.zeros((R,))

                if nd:
                    recall[t, k] = rc[-1]
                else:
                    recall[t, k] = 0

                # numpy is slow without cython optimization for accessing elements
                # use python array gets significant speed improvement
                pr = pr.tolist()
                q = q.tolist()

                for i in range(nd - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                inds = np.searchsorted(rc, p.recall_thresholds, side="left")
                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                except Exception:
                    # It seems this try/except is just a silly way to handle corner cases
                    pass
                precision[t, :, k] = np.array(q)

        return DatasetLevelEvaluationResult(
            params=p,
            counts=(T, R, K),
            precision=precision,
            recall=recall,
        )

    def accumulate_with_sg(self, evalImgs, imgIds, catIds):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """

        p = self.params

        T = len(p.iou_thresholds)
        K = len(catIds)

        # create dictionary for future indexing
        setK = set(catIds)
        setI = set(imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(catIds) if k in setK]
        i_list = [n for n, i in enumerate(imgIds) if i in setI]
        I0 = len(imgIds)

        precision = -torch.ones((T, K))
        recall = -torch.ones((T, K))

        # retrieve E at each category
        for k, k0 in enumerate(k_list):
            Nk = k0 * I0
            E = [evalImgs[Nk + i] for i in i_list]
            E: List[ImageLevelEvaluationResult] = [e for e in E if e is not None]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e.dtScores[0 : p.maxDets] for e in E])

            preds_matched = np.concatenate([e.dtMatches[:, 0 : p.maxDets] for e in E], axis=1)
            preds_to_ignore = np.concatenate([e.dtIgnore[:, 0 : p.maxDets] for e in E], axis=1)
            preds_scores = dtScores
            gtIg = np.concatenate([e.gtIgnore for e in E])
            gtCrowd = np.concatenate([e.gtIsCrowd for e in E])

            n_non_ignored_targets = np.count_nonzero(gtIg == 0)
            n_non_crowd = (gtCrowd == 0).sum()

            if n_non_ignored_targets == 0:
                continue

            _, cls_precision, cls_recall = compute_detection_metrics_per_cls(
                preds_matched=torch.from_numpy(preds_matched).moveaxis(0, 1) > 0,
                preds_to_ignore=torch.from_numpy(preds_to_ignore).moveaxis(0, 1) > 0,
                preds_scores=torch.from_numpy(preds_scores),
                n_targets=n_non_crowd,
                recall_thresholds=torch.from_numpy(p.recall_thresholds),
                score_threshold=0,
                device="cpu",
            )
            precision[:, k] = cls_precision
            recall[:, k] = cls_recall

        return DatasetLevelEvaluationResult(
            params=p,
            counts=(T, K),
            precision=precision.detach().cpu().numpy(),
            recall=recall.detach().cpu().numpy(),
        )
