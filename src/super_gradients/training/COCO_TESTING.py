import os
import json
from enum import Enum
from pathlib import Path

import torch
from PIL import Image
from typing import Callable, Union

import numpy as np
from torch.utils.data import SequentialSampler
from torch import nn
from tqdm import tqdm
from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

# from infery.inference.infery_manager import LoadedModel

import logging

import numpy as np
import time
from collections import defaultdict
import copy



class Params:
    def __init__(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)


class COCOeval:
    """
    A stripped-down version of COCOEval for easier debugging and better performance.
    """
    def __init__(self, cocoGt=None, cocoDt=None):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params()              # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        gts = np.array(self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)))
        dts = np.array(self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)))

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))
        self.params=p

        self._prepare()

        self.ious = {(imgId, catId): self.computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in p.catIds}

        self.evalImgs = [self.evaluateImg(imgId, catId)
                 for catId in p.catIds
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def iou(self, dt: np.ndarray, gt: np.ndarray, iscrowd: np.ndarray) -> np.ndarray:
        if len(dt) == 0 or len(gt) == 0:
            return np.zeros((0, 0))

        dt_area = dt[:, 2] * dt[:, 3]
        dt_right = dt[:, 0] + dt[:, 2]
        dt_top = dt[:, 1] + dt[:, 3]

        gt_area = gt[:, 2] * gt[:, 3]
        gt_right = gt[:, 0] + gt[:, 2]
        gt_top = gt[:, 1] + gt[:, 3]

        intersection_left = np.maximum(np.reshape(dt[:, 0], (-1, 1)), np.reshape(gt[:, 0], (1, -1)))
        intersection_right = np.minimum(np.reshape(dt_right, (-1, 1)), np.reshape(gt_right, (1, -1)))
        intersection_bottom = np.maximum(np.reshape(dt[:, 1], (-1, 1)), np.reshape(gt[:, 1], (1, -1)))
        intersection_top = np.minimum(np.reshape(dt_top, (-1, 1)), np.reshape(gt_top, (1, -1)))

        intersection_area = np.maximum(intersection_right - intersection_left, 0) * \
                            np.maximum(intersection_top - intersection_bottom, 0)
        union_area = np.reshape(dt_area, (-1, 1)) + np.logical_not(iscrowd) * (np.reshape(gt_area, (1, -1))
                                                                               - intersection_area)

        return intersection_area / union_area


    def computeIoU(self, imgId, catId):
        gt = self._gts[imgId,catId]
        dt = self._dts[imgId,catId]

        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > 100:
            dt=dt[0:100]

        g = np.array([g['bbox'] for g in gt])
        d = np.array([d['bbox'] for d in dt])

        # compute iou between each dt and gt region
        iscrowd = np.array([int(o['iscrowd']) for o in gt])
        # ious = maskUtils.iou(d,g,iscrowd)
        ious = self.iou(d,g,iscrowd)
        return ious

    def evaluateImg(self, imgId, catId):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        gt = self._gts[imgId,catId]
        dt = self._dts[imgId,catId]
        if len(gt) == 0 and len(dt) ==0:
            return None

        # sort dt highest score first, sort gt iscrowd last
        gtind = np.argsort([g['iscrowd'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:100]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['iscrowd'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds)
        precision   = -np.ones((T,R,K)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K))

        # create dictionary for future indexing
        _pe = self._paramsEval
        setK = set(_pe.catIds)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        # retrieve E at each category
        for k, k0 in enumerate(k_list):
            Nk = k0*I0
            E = [self.evalImgs[Nk + i] for i in i_list]
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e['dtScores'][0:100] for e in E])

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dtScores, kind='mergesort')

            dtm  = np.concatenate([e['dtMatches'][:,0:100] for e in E], axis=1)[:,inds]
            dtIg = np.concatenate([e['dtIgnore'][:,0:100]  for e in E], axis=1)[:,inds]
            gtIg = np.concatenate([e['gtIgnore'] for e in E])
            npig = np.count_nonzero(gtIg==0 )
            if npig == 0:
                continue
            tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp+tp+np.spacing(1))
                q  = np.zeros((R,))

                if nd:
                    recall[t,k] = rc[-1]
                else:
                    recall[t,k] = 0

                # numpy is slow without cython optimization for accessing elements
                # use python array gets significant speed improvement
                pr = pr.tolist(); q = q.tolist()

                for i in range(nd-1, 0, -1):
                    if pr[i] > pr[i-1]:
                        pr[i-1] = pr[i]

                inds = np.searchsorted(rc, p.recThrs, side='left')
                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                except:
                    pass
                precision[t,:,k] = np.array(q)
        self.eval = {
            'precision': precision
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize():
            iStr = ' {:<18} {} @[ IoU={:<9} ] = {:0.5f}'
            p = self.params
            titleStr = 'Average Precision'
            typeStr = '(AP)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1])

            # dimension of precision: [TxRxK]
            s = self.eval['precision']
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, mean_s))
            return mean_s
        if not self.eval:
            raise Exception('Please run accumulate() first')
        self.stats = _summarize()

    # Originally, COCOEval prints whenever __str__ is called, which is annoying during debug.
    # def __str__(self):
    #     self.summarize()


_COCO_91_INDEX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                  59,
                  60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89,
                  90]


class CocoMAP:
    def __init__(self, model: Union[nn.Module], post_prediction_callback: Callable, dataset_interface,
                 annotations_json_path: str) -> None:
        """
        :param model:                       A torch nn.Module or Infery Inferencer object to run inference on.
        :param post_prediction_callback:    A callable function for post-processing the output of the model (ie: NMS)
        :param dataset_interface:           A Deci Dataset Interface for loading the images and labels.
        :param annotations_json_path:       The path to the annotations json.
                                            NOTE: it's important the dataset interface and annotations_json_path be
                                            linked to the same dataset (ie val2017) otherwise the results can be wrong.
        Check examples/utils/calculate_coco_api_map.py for an example.
        """
        self.dataset_interface = dataset_interface
        self.annotations_json_path = annotations_json_path
        self._validate_model_type(model)
        self.model = model
        self.post_prediction_callback = post_prediction_callback

        if not os.path.exists(annotations_json_path):
            raise FileNotFoundError(f"File {annotations_json_path} was not found. "
                                    f"Probably because COCO is not deployed on server.")

        # converts the 80 classes of 80-index to 91-index (paper) for the CocoAPI
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        self.class_map = _COCO_91_INDEX

    def _undo_scale_letterbox(self, preds, orig_images_wh, cur_image_side):
        '''
        A utility function for unscaling the predictions wrt the original images sizes.
        '''
        for i in range(len(preds)):
            w, h = orig_images_wh[i]
            resize = cur_image_side / max(orig_images_wh[i])
            after_resize_w, after_resize_h = (int(w * resize), int(h * resize))
            scale = (w / after_resize_w, h / after_resize_h)
            pad = (cur_image_side - min(after_resize_w, after_resize_h)) / 2

            pads = (0, pad) if w > h else (pad, 0)  # pad for x y

            if preds[i] is not None:
                preds[i][:, [1, 3]] = preds[i][:, [1, 3]].clamp(min=0, max=cur_image_side)
                preds[i][:, [0, 2]] = preds[i][:, [0, 2]].clamp(min=0, max=cur_image_side)

            preds[i] = preds[i].detach().cpu().numpy() if preds[i] is not None else np.empty((0, 6))
            preds[i][:, :4] = (preds[i][:, :4] - (*pads, *pads)) * (*scale, *scale)
            preds[i][:, :4] = np.maximum(preds[i][:, :4],0)  # overrides negative coordinates with zero to avoid problems
        return preds

    def _update_coco_list(self, coco_list, outputs, image_ids):
        '''
        A utility function for processing the outputs for later calculations of statistics
        Example of appended object: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        '''
        for pred, image_id in zip(outputs, image_ids):
            for box in pred:
                coco_list.append({'image_id': int(image_id),
                                  'category_id': self.class_map[int(box[-1])],
                                  'bbox': [float(v) for v in box[:4]],
                                  'score': float(box[-2])})
        return coco_list

    def _xyxy_to_xywh(self, preds):
        for i in range(len(preds)):
            preds[i][:, 2] -= preds[i][:, 0]  # x2 to w
            preds[i][:, 3] -= preds[i][:, 1]  # y2 to h
        return preds

    def _predict(self, imgs):
        '''
        A utility function for allowing inference on both Infery Inferencer and torch nn.Module
        '''
        if self.infery_inference:
            return self.model.predict(imgs.cpu().numpy())
        else:
            with torch.no_grad():
                return self.model(imgs.to(self.device))

    def _validate_model_type(self, model):
        '''
        A utility function for validating the model is of a compatible type (torch nn.Module or Infery Inferencer) and
        setting the flag for the inference type.
        '''
        if isinstance(model, nn.Module):
            self.infery_inference = False
            self.device = next(model.parameters()).device
        # elif isinstance(model):
        #     self.infery_inference = True
        else:
            raise TypeError(f'Model type {type(model)} not supported')

    def _validate_val_loader(self, val_loader):
        assert isinstance(val_loader.sampler, SequentialSampler), "The function relies on the fact the input " \
                                                                  "data is not shuffled, so the sampler must be " \
                                                                  "SequentialSampler "
        # assert val_loader.dataset.sample_loading_method == 'default', "Default loading method = letterboxed, " \
        #                                                               "which is compatible with both infery and " \
        #                                                               "nn.module "

    def calculate_coco_map(self, output_json_path: str):
        """
        Calculates mAP using the COCO API.
        :param output_json_path:    The path to the json file where the eval results will be saved. If a pre-existing
                                    path is supplied, the function will not re-run the inference and proceed straight to processing the raw results
                                    that are saved in the json.
        :return eval:               The COCO eval object holding the statistics.
                                    For example: map050:095 = eval.stats[0], map050=eval.stats[1].
        """
        if os.path.exists(output_json_path):
            logging.warning('[WARNING] Found precalculated json file, taking results from there...')
        else:
            _, val_loader, _, _ = self.dataset_interface.get_data_loaders(batch_size_factor=1, num_workers=1)
            self._validate_val_loader(val_loader)
            batch_size = self.dataset_interface.dataset_params.val_batch_size
            coco_list = []
            for batch_i, (imgs, _, _) in enumerate(tqdm(val_loader)):
                outputs = self._predict(imgs)
                outputs = self.post_prediction_callback(outputs, self.device)
                paths = self.dataset_interface.valset.img_files[batch_i * batch_size: batch_i * batch_size + batch_size]
                paths = [Path(p) for p in paths]
                image_ids = [int(p.stem) if p.stem.isnumeric() else p.stem for p in paths]
                image_sizes = [Image.open(p).size for p in
                               paths]  # This is a lazy operation, doesn't actually load the image to memory
                outputs = self._undo_scale_letterbox(outputs, image_sizes,
                                                     self.dataset_interface.dataset_params.val_image_size)
                outputs = self._xyxy_to_xywh(outputs)
                coco_list = self._update_coco_list(coco_list, outputs, image_ids)
            with open(output_json_path, 'w') as f:
                json.dump(coco_list, f)

        anno = COCO(self.annotations_json_path)  # init annotations api
        pred = anno.loadRes(output_json_path)  # init predictions api
        eval = COCOeval(anno, pred)
        eval.params.imgIds = [int(Path(x).stem) for x in self.dataset_interface.valset.img_files]
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        return eval




# class COCOEvaluator:
#     """
#     COCO AP Evaluation class.  All the data in the val2017 dataset are processed
#     and evaluated by COCO API.
#     """
#
#     def __init__(
#         self,
#         dataloader,
#         img_size: int,
#         confthre: float,
#         nmsthre: float,
#         num_classes: int,
#         testdev: bool = False,
#     ):
#         """
#         Args:
#             dataloader (Dataloader): evaluate dataloader.
#             img_size: image size after preprocess. images are resized
#                 to squares whose shape is (img_size, img_size).
#             confthre: confidence threshold ranging from 0 to 1, which
#                 is defined in the config file.
#             nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
#             per_class_AP: Show per class AP during evalution or not. Default to False.
#             per_class_AR: Show per class AR during evalution or not. Default to False.
#         """
#         self.dataloader = dataloader
#         self.img_size = img_size
#         self.confthre = confthre
#         self.nmsthre = nmsthre
#         self.num_classes = num_classes
#         self.testdev = testdev
#         self.per_class_AP = per_class_AP
#         self.per_class_AR = per_class_AR
#
#     def evaluate(
#         self, model, distributed=False, half=False, trt_file=None,
#         decoder=None, test_size=None, return_outputs=False
#     ):
#         """
#         COCO average precision (AP) Evaluation. Iterate inference on the test dataset
#         and the results are evaluated by COCO API.
#         NOTE: This function will change training mode to False, please save states if needed.
#         Args:
#             model : model to evaluate.
#         Returns:
#             ap50_95 (float) : COCO AP of IoU=50:95
#             ap50 (float) : COCO AP of IoU=50
#             summary (sr): summary info of evaluation.
#         """
#         # TODO half to amp_test
#         tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
#         model = model.eval()
#         if half:
#             model = model.half()
#         ids = []
#         data_list = []
#         output_data = defaultdict()
#         progress_bar = tqdm
#
#         inference_time = 0
#         nms_time = 0
#         n_samples = max(len(self.dataloader) - 1, 1)
#
#         for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
#             progress_bar(self.dataloader)
#         ):
#             with torch.no_grad():
#                 imgs = imgs.type(tensor_type)
#
#                 # skip the last iters since batchsize might be not enough for batch inference
#                 is_time_record = cur_iter < len(self.dataloader) - 1
#                 if is_time_record:
#                     start = time.time()
#
#                 outputs = model(imgs)
#                 if decoder is not None:
#                     outputs = decoder(outputs, dtype=outputs.type())
#
#                 outputs = postprocess(
#                     outputs, self.num_classes, self.confthre, self.nmsthre
#                 )
#
#             data_list_elem, image_wise_data = self.convert_to_coco_format(
#                 outputs, info_imgs, ids, return_outputs=True)
#             data_list.extend(data_list_elem)
#             output_data.update(image_wise_data)
#
#         statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
#
#         eval_results = self.evaluate_prediction(data_list, statistics)
#
#         if return_outputs:
#             return eval_results, output_data
#         return eval_results
#
#     def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
#         data_list = []
#         image_wise_data = defaultdict(dict)
#         for (output, img_h, img_w, img_id) in zip(
#             outputs, info_imgs[0], info_imgs[1], ids
#         ):
#             if output is None:
#                 continue
#             output = output.cpu()
#
#             bboxes = output[:, 0:4]
#
#             # preprocessing: resize
#             scale = min(
#                 self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
#             )
#             bboxes /= scale
#             cls = output[:, 6]
#             scores = output[:, 4] * output[:, 5]
#
#             image_wise_data.update({
#                 int(img_id): {
#                     "bboxes": [box.numpy().tolist() for box in bboxes],
#                     "scores": [score.numpy().item() for score in scores],
#                     "categories": [
#                         self.dataloader.dataset.class_ids[int(cls[ind])]
#                         for ind in range(bboxes.shape[0])
#                     ],
#                 }
#             })
#
#             bboxes = xyxy2xywh(bboxes)
#
#             for ind in range(bboxes.shape[0]):
#                 label = self.dataloader.dataset.class_ids[int(cls[ind])]
#                 pred_data = {
#                     "image_id": int(img_id),
#                     "category_id": label,
#                     "bbox": bboxes[ind].numpy().tolist(),
#                     "score": scores[ind].numpy().item(),
#                     "segmentation": [],
#                 }  # COCO json format
#                 data_list.append(pred_data)
#
#         if return_outputs:
#             return data_list, image_wise_data
#         return data_list
#
#     def evaluate_prediction(self, data_dict, statistics):
#         annType = ["segm", "bbox", "keypoints"]
#
#         inference_time = statistics[0].item()
#         nms_time = statistics[1].item()
#         n_samples = statistics[2].item()
#
#         a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
#         a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)
#
#         time_info = ", ".join(
#             [
#                 "Average {} time: {:.2f} ms".format(k, v)
#                 for k, v in zip(
#                     ["forward", "NMS", "inference"],
#                     [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
#                 )
#             ]
#         )
#
#         info = time_info + "\n"
#
#         # Evaluate the Dt (detection) json comparing with the ground truth
#         if len(data_dict) > 0:
#             cocoGt = self.dataloader.dataset.coco
#             # TODO: since pycocotools can't process dict in py36, write data to json file.
#             if self.testdev:
#                 json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
#                 cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
#             else:
#                 _, tmp = tempfile.mkstemp()
#                 json.dump(data_dict, open(tmp, "w"))
#                 cocoDt = cocoGt.loadRes(tmp)
#             try:
#                 from yolox.layers import COCOeval_opt as COCOeval
#             except ImportError:
#                 from pycocotools.cocoeval import COCOeval
#
#             cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
#             cocoEval.evaluate()
#             cocoEval.accumulate()
#             cat_ids = list(cocoGt.cats.keys())
#             cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
#
#             return cocoEval, cocoEval.stats[0], cocoEval.stats[1], info
#         else:
#             return 0, 0, 0, info
#

    # ==============================



class CocoMAPV2:
    def __init__(self, model: Union[nn.Module], post_prediction_callback: Callable, dataset_interface,
                 annotations_json_path: str) -> None:
        """
        :param model:                       A torch nn.Module or Infery Inferencer object to run inference on.
        :param post_prediction_callback:    A callable function for post-processing the output of the model (ie: NMS)
        :param dataset_interface:           A Deci Dataset Interface for loading the images and labels.
        :param annotations_json_path:       The path to the annotations json.
                                            NOTE: it's important the dataset interface and annotations_json_path be
                                            linked to the same dataset (ie val2017) otherwise the results can be wrong.
        Check examples/utils/calculate_coco_api_map.py for an example.
        """
        self.dataset_interface = dataset_interface
        self.annotations_json_path = annotations_json_path
        self._validate_model_type(model)
        self.model = model
        self.post_prediction_callback = post_prediction_callback

        if not os.path.exists(annotations_json_path):
            raise FileNotFoundError(f"File {annotations_json_path} was not found. "
                                    f"Probably because COCO is not deployed on server.")

        # converts the 80 classes of 80-index to 91-index (paper) for the CocoAPI
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        self.class_map = _COCO_91_INDEX

    def _undo_scale_letterbox(self, preds, orig_images_wh, cur_image_side):
        '''
        A utility function for unscaling the predictions wrt the original images sizes.
        '''
        for i in range(len(preds)):
            w, h = orig_images_wh[i]
            resize = cur_image_side / max(orig_images_wh[i])
            after_resize_w, after_resize_h = (int(w * resize), int(h * resize))
            scale = (w / after_resize_w, h / after_resize_h)
            pad = (cur_image_side - min(after_resize_w, after_resize_h)) / 2

            pads = (0, pad) if w > h else (pad, 0)  # pad for x y

            if preds[i] is not None:
                preds[i][:, [1, 3]] = preds[i][:, [1, 3]].clamp(min=0, max=cur_image_side)
                preds[i][:, [0, 2]] = preds[i][:, [0, 2]].clamp(min=0, max=cur_image_side)

            preds[i] = preds[i].detach().cpu().numpy() if preds[i] is not None else np.empty((0, 6))
            preds[i][:, :4] = (preds[i][:, :4] - (*pads, *pads)) * (*scale, *scale)
            preds[i][:, :4] = np.maximum(preds[i][:, :4],0)  # overrides negative coordinates with zero to avoid problems
        return preds

    def _update_coco_list(self, coco_list, outputs, image_ids):
        '''
        A utility function for processing the outputs for later calculations of statistics
        Example of appended object: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        '''
        for pred, image_id in zip(outputs, image_ids):
            for box in pred:
                coco_list.append({'image_id': int(image_id),
                                  'category_id': self.class_map[int(box[-1])],
                                  'bbox': [float(v) for v in box[:4]],
                                  'score': float(box[-2])})
        return coco_list

    def _xyxy_to_xywh(self, preds):
        for i in range(len(preds)):
            preds[i][:, 2] -= preds[i][:, 0]  # x2 to w
            preds[i][:, 3] -= preds[i][:, 1]  # y2 to h
        return preds

    def _predict(self, imgs):
        '''
        A utility function for allowing inference on both Infery Inferencer and torch nn.Module
        '''
        if self.infery_inference:
            return self.model.predict(imgs.cpu().numpy())
        else:
            with torch.no_grad():
                return self.model(imgs.to(self.device))

    def _validate_model_type(self, model):
        '''
        A utility function for validating the model is of a compatible type (torch nn.Module or Infery Inferencer) and
        setting the flag for the inference type.
        '''
        if isinstance(model, nn.Module):
            self.infery_inference = False
            self.device = next(model.parameters()).device
        # elif isinstance(model):
        #     self.infery_inference = True
        else:
            raise TypeError(f'Model type {type(model)} not supported')

    def _validate_val_loader(self, val_loader):
        self.val_loader = val_loader
        assert isinstance(val_loader.sampler, SequentialSampler), "The function relies on the fact the input " \
                                                                  "data is not shuffled, so the sampler must be " \
                                                                  "SequentialSampler "

    def calculate_coco_map(self, output_json_path: str):
        """
        Calculates mAP using the COCO API.
        :param output_json_path:    The path to the json file where the eval results will be saved. If a pre-existing
                                    path is supplied, the function will not re-run the inference and proceed straight to processing the raw results
                                    that are saved in the json.
        :return eval:               The COCO eval object holding the statistics.
                                    For example: map050:095 = eval.stats[0], map050=eval.stats[1].
        """
        if os.path.exists(output_json_path):
            logging.warning('[WARNING] Found precalculated json file, taking results from there...')
        else:
            _, val_loader, _, _ = self.dataset_interface.get_data_loaders(batch_size_factor=1, num_workers=1)
            self._validate_val_loader(val_loader)
            batch_size = self.dataset_interface.dataset_params.val_batch_size
            coco_list = []
            all_image_ids = []

            from torch.utils.data._utils.collate import default_collate

            # Use default collate fonction to include every item returned by the dataloader in the batch
            val_loader.collate_fn, keep_collate_fn = default_collate, val_loader.collate_fn

            for batch_i, (imgs, _, _, info_imgs, ids) in enumerate(tqdm(val_loader)):
                outputs = self._predict(imgs)
                outputs = self.post_prediction_callback(outputs, self.device)
                output_formated = self.convert_to_coco_format(outputs, info_imgs, ids)
                coco_list.extend(output_formated)
                all_image_ids.extend(ids)

            with open(output_json_path, 'w') as f:
                json.dump(coco_list, f)
            val_loader.collate_fn = keep_collate_fn

        anno = COCO(self.annotations_json_path)  # init annotations api
        pred = anno.loadRes(output_json_path)  # init predictions api
        eval = COCOeval(anno, pred)
        eval.params.imgIds = all_image_ids
        all_image_ids
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        return eval

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue

            output = output.cpu()
            bboxes, scores, cls = output[:, 0:4], output[:, 4], output[:, 5]

            # preprocessing: resize
            image_size = self.dataset_interface.dataset_params.val_image_size
            scale = min(image_size/ float(img_h), image_size / float(img_w))
            bboxes /= scale

            def xyxy2xywh(bboxes):
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
                bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
                return bboxes

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.class_map[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }
                data_list.append(pred_data)

        return data_list






# ==============================
import argparse
import os
import tempfile

import torch

from super_gradients.training.datasets.dataset_interfaces import CoCoDetectionDatasetInterface #CocoDetectionDatasetInterfaceV2
# from super_gradients.training.models.detection_models.yolov5 import YoloV5PostPredictionCallback
from super_gradients.training.utils import checkpoint_utils
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients import ARCHITECTURES
from super_gradients.training.utils.utils import HpmStruct

# parser = argparse.ArgumentParser(description="Calculate mAP for a checkpoint using COCO API.")
# parser.add_argument(
#     'ckpt_path', default="oo"
# )
# parser.add_argument(
#     '--architecture', '-a',
#     default='yolo_v5s'
# )
# parser.add_argument(
#     '--batch-size', '-b',
#     type=int,
#     default=1
# )
# parser.add_argument(
#     '--image-size', '-i',
#     default=640
# )
# parser.add_argument(
#     '--nms-conf',
#     type=float,
#     default=0.001
# )
# parser.add_argument(
#     '--nms-iou',
#     type=float,
#     default=0.65
# )
# parser.add_argument(
#     '--predictions-file', "-f",
#     help='Optional path for the predictions file. '
#          'If the file already exists then predictions will be loaded from it. '
#          'Otherwise, predictions will be written to it.'
# )
import super_gradients
from omegaconf import DictConfig
import hydra
import pkg_resources
from super_gradients.training.sg_model.sg_model import SgModel

super_gradients.init_trainer()

@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""))
def instantiate_dataset(cfg: DictConfig):


    from super_gradients.training.metrics.detection_metrics import DetectionMetrics
    from super_gradients.training.models.detection_models.yolov5_base import YoloV5PostPredictionCallback
    post_prediction_callback = YoloV5PostPredictionCallback()#iou=0.65, conf=0.01)

    #
    # # INSTANTIATE ALL OBJECTS IN CFG
    # cfg = hydra.utils.instantiate(cfg)
    # coco_dataset = CocoDetectionDatasetInterfaceV2(dataset_params=dict(cfg["dataset_params"]))

    cfg = hydra.utils.instantiate(cfg)
    cfg.sg_model.connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)
    cfg.sg_model.build_model(cfg.architecture, arch_params=cfg.arch_params, checkpoint_params=cfg.checkpoint_params)
    trainer = cfg.sg_model

    model = trainer.net
    model.eval()
    res = trainer.test(
        test_loader=cfg.dataset_interface.val_loader,
        test_metrics_list=[DetectionMetrics(post_prediction_callback=post_prediction_callback,
                                            num_cls=80, normalize_targets=True)],
        metrics_progress_verbose=False
    )


    # dataset_params = {"train_batch_size": 1, "val_batch_size": 1,
    #                   "train_image_size": 640,
    #                   "val_image_size": 640,
    #                   "val_sample_loading_method": "default"}
    # coco_dataset = CoCoDetectionDatasetInterface(dataset_params=dataset_params, with_crowd=True)
    # coco_pretrained_ckpt_params = {"pretrained_weights": "coco"}
    #
    # trainer = SgModel('yolo_v5m_coco')
    # trainer.connect_dataset_interface(coco_dataset)
    # trainer.build_model("yolo_v5m", checkpoint_params=coco_pretrained_ckpt_params)
    #
    # res = trainer.test(
    #     test_loader=coco_dataset.val_loader,
    #     test_metrics_list=[DetectionMetrics(post_prediction_callback=post_prediction_callback,
    #                                         num_cls=-1, normalize_targets=False)],
    #     metrics_progress_verbose=False
    # )


    print("res: ", res)
    model = trainer.net
    model.eval()

    anno_json = '/data/coco/annotations/instances_val2017.json'
    coco_map_tool = CocoMAPV2(model=model,
                            post_prediction_callback=post_prediction_callback,
                            dataset_interface=cfg.dataset_interface,
                            annotations_json_path=anno_json)
    import random
    predictions_file = f"pred_yolo{random.random()}"

    if predictions_file is None:
        tmpdir = tempfile.TemporaryDirectory()
        pred_json = os.path.join(tmpdir.name, f'predictions.json')
        print(f"Temporary path for predictions is {pred_json}.")
    else:
        tmpdir = None
        pred_json = predictions_file

    try:
        map_result = coco_map_tool.calculate_coco_map(output_json_path=pred_json)
        print(f"mAP@0.5:0.95: {map_result.stats[0]}")
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
    print("SS")
#
#
# class IouThreshold(tuple, Enum):
#     MAP_05 = (0.5, 0.5)
#     MAP_05_TO_095 = (0.5, 0.95)
#
# from dataclasses import dataclass
#
#
# class BBoxFormat(Enum):
#     XYXY = "XYXY"
#     XYWH = "XYWH"
#     CXCYWH = "XYXY"
#
# class Normalization(Enum):
#     NONE = "Not normalized (image size)"
#     NORMALIZED_01 = "Normalized between 0 to 1"
#
# @dataclass
# class BBoxInfo:
#     format: BBoxFormat
#     normalization: Normalization
#
#
# BBoxInfo(BBoxFormat.CXCYWH, Normalization.NORMALIZED_01)
#
#
if __name__ == "__main__":
    instantiate_dataset()

