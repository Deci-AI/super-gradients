from typing import List, Any, Tuple

import torch
import torchvision

from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from torch import Tensor


class PPYoloEPostPredictionCallback(DetectionPostPredictionCallback):
    """Non-Maximum Suppression (NMS) module"""

    def __init__(
        self,
        *,
        score_threshold: float,
        nms_threshold: float,
        nms_top_k: int,
        max_predictions: int,
        multi_label_per_box: bool = True,
        class_agnostic_nms: bool = False,
    ):
        """
        :param score_threshold:     Predictions confidence threshold.
                                    Predictions with score lower than score_threshold will not participate in Top-K & NMS
        :param nms_threshold:       IoU threshold for NMS step.
        :param nms_top_k:           Number of predictions participating in NMS step
        :param max_predictions:     Maximum number of boxes to return after NMS step
        :param multi_label_per_box: Controls whether to decode multiple labels per box.
                                    True - each anchor can produce multiple labels of different classes
                                           that pass confidence threshold check (default).
                                    False - each anchor can produce only one label of the class with the highest score.
        """
        super(PPYoloEPostPredictionCallback, self).__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.max_predictions = max_predictions
        self.multi_label_per_box = multi_label_per_box
        self.class_agnostic_nms = class_agnostic_nms

    @torch.no_grad()
    def forward(self, outputs: Any, device: str = None) -> List[List[Tensor]]:
        """

        :param outputs: Outputs of model's forward() method
        :param device:  (Deprecated) Not used anymore, exists only for sake of keeping the same interface as in the parent class.
                        Will be removed in the SG 3.7.0.
                        A device parameter in case we want to move tensors to a specific device.
        :return:        List of lists of tensors of shape [Ni, 6] where Ni is the number of detections in i-th image.
                        Format of each row is [x1, y1, x2, y2, confidence, class]
        """
        nms_result = []
        predictions = self._get_decoded_predictions_from_model_output(outputs)

        for pred_bboxes, pred_scores in zip(*predictions):
            # Cast to float to avoid lack of fp16 support in torchvision.ops.boxes.batched_nms when doing CPU inference
            pred_bboxes = pred_bboxes.float()  # [Anchors, 4]
            pred_scores = pred_scores.float()  # [Anchors, C]

            # Filter all predictions by self.score_threshold
            if self.multi_label_per_box:
                i, j = (pred_scores > self.score_threshold).nonzero(as_tuple=False).T
                pred_bboxes = pred_bboxes[i]
                pred_cls_conf = pred_scores[i, j]
                pred_cls_label = j[:]

            else:
                pred_cls_conf, pred_cls_label = torch.max(pred_scores, dim=1)
                conf_mask = pred_cls_conf >= self.score_threshold

                pred_cls_conf = pred_cls_conf[conf_mask]
                pred_cls_label = pred_cls_label[conf_mask]
                pred_bboxes = pred_bboxes[conf_mask, :]

            # Filter all predictions by self.nms_top_k
            if pred_cls_conf.size(0) > self.nms_top_k:
                topk_candidates = torch.topk(pred_cls_conf, k=self.nms_top_k, largest=True)
                pred_cls_conf = pred_cls_conf[topk_candidates.indices]
                pred_cls_label = pred_cls_label[topk_candidates.indices]
                pred_bboxes = pred_bboxes[topk_candidates.indices, :]

            # NMS
            if self.class_agnostic_nms:
                idx_to_keep = torchvision.ops.boxes.nms(pred_bboxes, pred_cls_conf, iou_threshold=self.nms_threshold)
            else:
                idx_to_keep = torchvision.ops.boxes.batched_nms(boxes=pred_bboxes, scores=pred_cls_conf, idxs=pred_cls_label, iou_threshold=self.nms_threshold)

            pred_cls_conf = pred_cls_conf[idx_to_keep].unsqueeze(-1)
            pred_cls_label = pred_cls_label[idx_to_keep].unsqueeze(-1)
            pred_bboxes = pred_bboxes[idx_to_keep, :]

            #  nx6 (x1, y1, x2, y2, confidence, class) in pixel units
            final_boxes = torch.cat([pred_bboxes, pred_cls_conf, pred_cls_label], dim=1)  # [N,6]

            nms_result.append(final_boxes)

        return self._filter_max_predictions(nms_result)

    def _get_decoded_predictions_from_model_output(self, outputs: Any) -> Tuple[Tensor, Tensor]:
        """
        Get the decoded predictions from the PPYoloE/YoloNAS output.
        Depending on the model regime (train/eval) the output format may differ so this method picks the right output.

        :param outputs: Model's forward() return value
        :return:        Tuple of (bboxes, scores) of shape [B, Anchors, 4], [B, Anchors, C]
        """
        if isinstance(outputs, tuple) and len(outputs) == 2:
            if torch.is_tensor(outputs[0]) and torch.is_tensor(outputs[1]) and outputs[0].shape[1] == outputs[1].shape[1] and outputs[0].shape[2] == 4:
                # This path happens when we are using traced model or ONNX model without postprocessing for inference.
                predictions = outputs
            else:
                # First is model predictions, second element of tuple is logits for loss computation
                predictions = outputs[0]
        else:
            raise ValueError(f"Unsupported output format: {outputs}")

        return predictions

    def _filter_max_predictions(self, res: List) -> List:
        res[:] = [im[: self.max_predictions] if (im is not None and im.shape[0] > self.max_predictions) else im for im in res]

        return res
