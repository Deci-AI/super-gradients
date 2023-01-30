# This file contains modified version of COCOEval class from pycocotools library,
# which is used to compute mAP metric for pose estimation task.
#
# The changes are:
# - added support for user-defined maxDet parameter (It was hard-coded to 20 in original implementation)
# - removed print statements that output uninformative information
from typing import Union, List

import numpy as np
import datetime

# import time
from collections import defaultdict
import copy


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
    def __init__(self, cocoGt=None, cocoDt=None):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params()  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        p = self.params

        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        self.ious = {(imgId, catId): self.computeOksV2(imgId, catId) for imgId in p.imgIds for catId in catIds}

        maxDet = p.maxDets[-1]
        self.evalImgs = [self.evaluateImg(imgId, catId, areaRng, maxDet) for catId in catIds for areaRng in p.areaRng for imgId in p.imgIds]
        self._paramsEval = copy.deepcopy(self.params)

    def computeOksV2(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        groundtruths = self._gts[imgId, catId]
        predictions = self._dts[imgId, catId]

        # Note this unpacking will be removed once we get rid of COCO format
        pred_keypoints = np.array([np.array(dt["keypoints"]).reshape(-1, 3) for dt in predictions])
        pred_scores = np.array([dt["score"] for dt in predictions])

        gt_keypoints_with_visibility = np.array([np.array(gt["keypoints"]).reshape(-1, 3) for gt in groundtruths])
        gt_joints = gt_keypoints_with_visibility[:, :, 0:2] if len(gt_keypoints_with_visibility) else []
        gt_keypoint_visibility = gt_keypoints_with_visibility[:, :, 2] if len(gt_keypoints_with_visibility) else []
        gt_areas = np.array([gt["area"] for gt in groundtruths])
        gt_bboxes = np.array([gt["bbox"] for gt in groundtruths])
        # gt_areas = None
        # gt_bboxes = None

        ious = computeKeypointsIoU(
            pred_joints=pred_keypoints,
            pred_scores=pred_scores,
            gt_joints=gt_joints,
            gt_keypoint_visibility=gt_keypoint_visibility,
            gt_areas=gt_areas,
            gt_bboxes=gt_bboxes,
            max_dets=p.maxDets[-1],
            sigmas=p.kpt_oks_sigmas,
        )
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
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
        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        # print("Accumulating evaluation results...")
        # tic = time.time()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
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
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except Exception:
                            # It seems this try/except is just a silly way to handle corner cases
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        # toc = time.time()
        # print("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else "{:0.2f}".format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            # stats[0] = _summarize(1, maxDets=20)
            # stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            # stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            # stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            # stats[4] = _summarize(1, maxDets=20, areaRng='large')
            # stats[5] = _summarize(0, maxDets=20)
            # stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            # stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            # stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            # stats[9] = _summarize(0, maxDets=20, areaRng='large')
            # Edited
            stats[0] = _summarize(1, maxDets=self.params.maxDets[0])
            stats[1] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.5)
            stats[2] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.75)
            stats[3] = _summarize(1, maxDets=self.params.maxDets[0], areaRng="medium")
            stats[4] = _summarize(1, maxDets=self.params.maxDets[0], areaRng="large")
            stats[5] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=0.5)
            stats[7] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=0.75)
            stats[8] = _summarize(0, maxDets=self.params.maxDets[0], areaRng="medium")
            stats[9] = _summarize(0, maxDets=self.params.maxDets[0], areaRng="large")

            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")

        self.stats = _summarizeKps()

    def __str__(self):
        self.summarize()


class Params:
    """
    Params for coco evaluation api
    """

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0

    def __init__(self):
        self.setKpParams()
