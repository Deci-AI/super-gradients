from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from super_gradients.training.utils.detection_utils import calculate_bbox_iou_matrix
from super_gradients.training.utils.ssd_utils import DefaultBoxes


class HardMiningCrossEntropyLoss(_Loss):
    """
    L_cls = [CE of all positives] + [CE of the hardest backgrounds]
    where the second term is built from [3 * positive pairs] background cells with the highest CE
    (the hardest background cells)
    """

    def __init__(self, neg_pos_ratio):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.ce = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred_labels, target_labels):
        mask = target_labels > 0  # not background
        pos_num = mask.sum(dim=1)

        # HARD NEGATIVE MINING
        con = self.ce(pred_labels, target_labels)

        # POSITIVE MASK WILL NOT BE SELECTED
        # set 0. loss for all positive objects, leave the loss where the object is background
        con_neg = con.clone()
        con_neg[mask] = 0
        # sort background cells by CE loss value (bigger_first)
        _, con_idx = con_neg.sort(dim=1, descending=True)
        # restore cells order, get each cell's order (rank) in CE loss sorting
        _, con_rank = con_idx.sort(dim=1)

        # NUMBER OF NEGATIVE THREE TIMES POSITIVE
        neg_num = torch.clamp(self.neg_pos_ratio * pos_num, max=mask.size(1)).unsqueeze(-1)
        # for each image into neg mask we'll take (3 * positive pairs) background objects with the highest CE
        neg_mask = con_rank < neg_num

        closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)
        return closs


class SigmoidFocalClassificationLoss(_Loss):

    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_labels, target_labels):
        pred_labels = pred_labels.permute(0, 2, 1)
        target_labels_onehot = F.one_hot(target_labels, num_classes=pred_labels.shape[-1]).float()
        per_entry_bce = self.bce(pred_labels, target_labels_onehot)
        with torch.no_grad():
            prediction_probs = torch.sigmoid(pred_labels)
        p_t = target_labels_onehot * prediction_probs + (1 - target_labels_onehot) * (1 - prediction_probs)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        alpha_weight_factor = (target_labels_onehot * self.alpha + (1 - target_labels_onehot) * (1 - self.alpha))
        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_bce
        return focal_cross_entropy_loss.mean(dim=-1).sum(dim=1)


class SSDLoss(_Loss):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels

    L = (2 - alpha) * L_l1 + alpha * L_cls, where
        * L_cls is either HardMiningCrossEntropyLoss or SigmoidFocalClassificationLoss
        * L_l1 = [SmoothL1Loss for all positives]
    """

    def __init__(self, dboxes: DefaultBoxes, alpha: float = 1.0, iou_thresh: float = 0.5, neg_pos_ratio: float = 3.,
                 rebalance=False, sigmoid_focal=False):
        super(SSDLoss, self).__init__()
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh
        self.alpha = alpha
        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False)

        if sigmoid_focal:
            self.con_loss = SigmoidFocalClassificationLoss(0.75, 2.)
        else:
            self.con_loss = HardMiningCrossEntropyLoss(neg_pos_ratio)
        self.iou_thresh = iou_thresh
        self.rebalance = rebalance

    def _norm_relative_bbox(self, loc):
        """
        convert bbox locations into relative locations (relative to the dboxes)
        :param loc a tensor of shape [batch, 4, num_boxes]
        """
        gxy = ((loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]) / self.scale_xy
        gwh = (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log() / self.scale_wh
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def match_dboxes(self, targets):
        """
        creates tensors with target boxes and labels for each dboxes, so with the same len as dboxes.

        * Each GT is assigned with a grid cell with the highest IoU, this creates a pair for each GT and some cells;
        * The rest of grid cells are assigned to a GT with the highest IoU, assuming it's > self.iou_thresh;
          If this condition is not met the grid cell is marked as background

        GT-wise: one to many
        Grid-cell-wise: one to one

        :param targets: a tensor containing the boxes for a single image;
                        shape [num_boxes, 6] (image_id, label, x, y, w, h)
        :return:        two tensors
                        boxes - shape of dboxes [4, num_dboxes] (x,y,w,h)
                        labels - sahpe [num_dboxes]
        """
        device = targets.device
        each_cell_target_locations = self.dboxes.data.clone().squeeze()
        each_cell_target_labels = torch.zeros((self.dboxes.data.shape[2])).to(device)

        if len(targets) > 0:
            target_boxes = targets[:, 2:]
            target_labels = targets[:, 1]
            ious = calculate_bbox_iou_matrix(target_boxes, self.dboxes.data.squeeze().T, x1y1x2y2=False)

            # one best GT for EACH cell (does not guarantee that all GTs will be used)
            best_target_per_cell, best_target_per_cell_index = ious.max(0)

            # one best grid cell for EACH target
            best_cell_per_target, best_cell_per_target_index = ious.max(1)
            # make sure EACH target has a grid cell assigned
            best_target_per_cell_index[best_cell_per_target_index] = torch.arange(len(targets)).to(device)
            best_target_per_cell[best_cell_per_target_index] = 2.

            mask = best_target_per_cell > self.iou_thresh
            each_cell_target_locations[:, mask] = target_boxes[best_target_per_cell_index[mask]].T
            each_cell_target_labels[mask] = target_labels[best_target_per_cell_index[mask]] + 1

        return each_cell_target_locations, each_cell_target_labels

    def forward(self, predictions: Tuple, targets):
        """
        Compute the loss
            :param predictions - predictions tensor coming from the network,
            tuple with shapes ([Batch Size, 4, num_dboxes], [Batch Size, num_classes + 1, num_dboxes])
            were predictions have logprobs for background and other classes
            :param targets - targets for the batch. [num targets, 6] (index in batch, label, x,y,w,h)
        """
        batch_target_locations = []
        batch_target_labels = []
        (ploc, plabel) = predictions
        targets = targets.to(self.dboxes.device)
        for i in range(ploc.shape[0]):
            target_locations, target_labels = self.match_dboxes(targets[targets[:, 0] == i])
            batch_target_locations.append(target_locations)
            batch_target_labels.append(target_labels)
        batch_target_locations = torch.stack(batch_target_locations)
        batch_target_labels = torch.stack(batch_target_labels).type(torch.long)

        mask = batch_target_labels > 0  # not background
        pos_num = mask.sum(dim=1)

        vec_gd = self._norm_relative_bbox(batch_target_locations)

        # SUM ON FOUR COORDINATES, AND MASK
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float() * sl1).sum(dim=1)

        closs = self.con_loss(plabel, batch_target_labels)

        # AVOID NO OBJECT DETECTED
        if self.rebalance:
            with torch.no_grad():
                wt_loc = closs / (1e-7 + sl1)
        else:
            wt_loc = 1.

        total_loss = (2 - self.alpha) * sl1 * wt_loc + self.alpha * closs
        num_mask = (pos_num > 0).float()  # a mask with 0 for images that have no positive pairs at all
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # normalize by the number of positive pairs

        return ret, torch.cat((sl1.mean().unsqueeze(0), closs.mean().unsqueeze(0), ret.unsqueeze(0))).detach()
