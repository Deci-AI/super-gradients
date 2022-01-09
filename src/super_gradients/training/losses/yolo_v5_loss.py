from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from super_gradients.training.losses.focal_loss import FocalLoss
from super_gradients.training.utils.detection_utils import calculate_bbox_iou_elementwise, Anchors


class YoLoV5DetectionLoss(_Loss):
    """
    Calculate YOLO V5 loss:
    L = L_objectivness + L_boxes + L_classification
    """

    def __init__(self, anchors: Anchors,
                 cls_pos_weight: Union[float, List[float]] = 1.0, obj_pos_weight: float = 1.0,
                 obj_loss_gain: float = 1.0, box_loss_gain: float = 0.05, cls_loss_gain: float = 0.5,
                 focal_loss_gamma: float = 0.0,
                 cls_objectness_weights: Union[List[float], torch.Tensor] = None, anchor_threshold=4.0):
        """
        :param anchors:                 the anchors of the model (same anchors used for training)
        :param cls_pos_weight:          pos_weight for BCE in L_classification,
                                        can be one value for all positives or a list of weights for each class
        :param obj_pos_weight:          pos_weight for BCE in L_objectivness
        :param obj_loss_gain:           coef for L_objectivness
        :param box_loss_gain:           coef for L_boxes
        :param cls_loss_gain:           coef for L_classification
        :param focal_loss_gamma:        gamma for a focal loss, 0 to train with a usual BCE
        :param cls_objectness_weights:  class-based weight for L_objectivness that will be applied in each cell that
                                        has a GT assigned to it.
                                        Note: default weight for objectness loss in each cell is 1.
        :param anchor_threshold:                ratio defining a size range of an appropriate anchor.
        """
        super(YoLoV5DetectionLoss, self).__init__()

        self.cls_pos_weight = cls_pos_weight
        self.obj_pos_weight = obj_pos_weight
        self.obj_loss_gain = obj_loss_gain
        self.box_loss_gain = box_loss_gain
        self.cls_loss_gain = cls_loss_gain
        self.focal_loss_gamma = focal_loss_gamma

        self.anchor_threshold = anchor_threshold

        self.anchors = anchors

        self.cls_obj_weights = cls_objectness_weights
        if isinstance(cls_objectness_weights, list):
            self.cls_obj_weights = torch.nn.Parameter(torch.tensor(cls_objectness_weights))

    def forward(self, model_output, targets):
        if isinstance(model_output, tuple) and len(model_output) == 2:
            # in test/eval mode the Yolo v5 model output a tuple where the second item is the raw predictions
            _, predictions = model_output
        else:
            predictions = model_output

        return self.compute_loss(predictions, targets)

    def build_targets(self, predictions: List[torch.Tensor], targets: torch.Tensor) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor]], List[torch.Tensor]]:
        """
        Assign targets to anchors to use in L_boxes & L_classification calculation:
            * each target can be assigned to a few anchors,
            all anchors that are within [1/self.anchor_threshold, self.anchor_threshold] times target size range
            * each anchor can be assigned to a few targets

        :param predictions:         Yolo predictions
        :param targets:             ground truth targets
        :return:                    each of 4 outputs contains one element for each Yolo output,
                                    correspondences are raveled over the whole batch and all anchors:
                                        * classes of the targets;
                                        * boxes of the targets;
                                        * image id in a batch, anchor id, grid y, grid x coordinates;
                                        * anchor sizes.
                                    All the above can be indexed in parallel to get the selected correspondences
        """
        num_anchors, num_targets = self.anchors.num_anchors, targets.shape[0]
        target_classes, target_boxes, indices, anchors = [], [], [], []

        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        anchor_indices = torch.arange(num_anchors, device=targets.device)
        anchor_indices = anchor_indices.float().view(num_anchors, 1).repeat(1, num_targets)
        # repeat all targets for each anchor and append a corresponding anchor index
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), anchor_indices[:, :, None]), 2)

        bias = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            ], device=targets.device).float() * bias  # offsets

        for i in range(self.anchors.detection_layers_num):
            anch = self.anchors.anchors[i]
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Convert target coordinates from [0, 1] range to coordinates in [0, GridY], [0, GridX] ranges
            t = targets * gain
            if num_targets:
                # Match: filter targets by anchor size ratio
                r = t[:, :, 4:6] / anch[:, None]  # wh ratio
                filtered_targets_ids = torch.max(r, 1. / r).max(2)[0] < self.anchor_threshold  # compare
                t = t[filtered_targets_ids]

                # Find coordinates of targets on a grid
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < bias) & (gxy > 1.)).T
                l, m = ((gxi % 1. < bias) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # prevent coordinates from going out of bounds
            gi, gj = gi.clamp_(0, gain[2] - 1), gj.clamp_(0, gain[3] - 1)

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            target_boxes.append(torch.cat((gxy - gij, gwh), 1))  # box
            anchors.append(anch[a])  # anchors
            target_classes.append(c)  # class

        return target_classes, target_boxes, indices, anchors

    def compute_loss(self, predictions: List[torch.Tensor], targets: torch.Tensor, giou_loss_ratio: float = 1.0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        L = L_objectivness + L_boxes + L_classification
        where:
            * L_boxes and L_classification are calculated only between anchors and targets that suit them;
            * L_objectivness is calculated on all anchors.

        L_classification:
            for anchors that have suitable ground truths in their grid locations add BCEs
            to force max probability for each GT class in a multi-label way
            Coef: self.cls_loss_gain
        L_boxes:
            for anchors that have suitable ground truths in their grid locations
            add (1 - IoU), IoU between a predicted box and each GT box, force maximum IoU
            Coef: self.box_loss_gain
        L_objectness:
            for each anchor add BCE to force a prediction of (1 - giou_loss_ratio) + giou_loss_ratio * IoU,
            IoU between a predicted box and random GT in it
            Coef: self.obj_loss_gain, loss from each YOLO grid is additionally multiplied by balance = [4.0, 1.0, 0.4]
                  to balance different contributions coming from different numbers of grid cells

        :param predictions:     output from all Yolo levels, each of shape
                                [Batch x Num_Anchors x GridSizeY x GridSizeX x (4 + 1 + Num_classes)]
        :param targets:         [Num_targets x (4 + 2)], values on dim 1 are: image id in a batch, class, box x y w h
        :param giou_loss_ratio: a coef in L_objectness defining what should be predicted as objecness
                                in a call with a target: can be a value in [IoU, 1] range
        :return:                loss, all losses separately in a detached tensor
        """
        device = targets.device
        loss_classification, loss_boxes, loss_objectivness = \
            torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        target_classes, target_boxes, indices, anchors = self.build_targets(predictions, targets)  # targets

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.cls_pos_weight])).to(device)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.obj_pos_weight]), reduction='none').to(device)

        # Focal loss
        if self.focal_loss_gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, self.focal_loss_gamma), FocalLoss(BCEobj, self.focal_loss_gamma)

        # Losses
        num_targets = 0
        num_predictions = len(predictions)
        balance = [4.0, 1.0, 0.4] if num_predictions == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        for i, prediction in enumerate(predictions):  # layer index, layer predictions
            image, anchor, grid_y, grid_x = indices[i]
            target_obj = torch.zeros_like(prediction[..., 0], device=device)
            weight_obj = torch.ones_like(prediction[..., 0], device=device)

            n = image.shape[0]  # number of targets
            if n:
                num_targets += n  # cumulative targets
                ps = prediction[image, anchor, grid_y, grid_x]  # prediction subset corresponding to targets

                # Boxes loss
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = calculate_bbox_iou_elementwise(pbox.T, target_boxes[i], x1y1x2y2=False, CIoU=True)
                loss_boxes += (1.0 - iou).mean()  # iou loss

                # Objectness loss target
                target_obj[image, anchor, grid_y, grid_x] = \
                    (1.0 - giou_loss_ratio) + giou_loss_ratio * iou.detach().clamp(0).type(target_obj.dtype)
                # Weights for weighted objectness
                if self.cls_obj_weights is not None:
                    # NOTE: for grid cells that have a few ground truths with different classes assigned to them
                    # objectness weight will be picked randomly from one of these classes
                    weight_obj[image, anchor, grid_y, grid_x] = self.cls_obj_weights[target_classes[i]]

                # Classification loss
                if ps.shape[1] > 6:   # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], 0, device=device)  # targets
                    t[range(n), target_classes[i]] = 1
                    loss_classification += BCEcls(ps[:, 5:], t)  # BCE

            # Objectness loss
            loss_obj_cur_head = BCEobj(prediction[..., 4], target_obj)
            loss_obj_cur_head = torch.sum(loss_obj_cur_head * weight_obj / torch.sum(weight_obj))
            loss_objectivness += loss_obj_cur_head * balance[i]  # obj loss

        batch_size = prediction.shape[0]  # batch size

        loss = loss_boxes * self.box_loss_gain + loss_objectivness * self.obj_loss_gain + loss_classification * self.cls_loss_gain
        # IMPORTANT: box, obj and cls loss are logged scaled by gain in ultralytics
        # and are logged unscaled in our codebase
        return loss * batch_size, torch.cat((loss_boxes, loss_objectivness, loss_classification, loss)).detach()
