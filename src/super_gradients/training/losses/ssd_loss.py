from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from super_gradients.training.utils.detection_utils import calculate_bbox_iou_matrix
from super_gradients.training.utils.ssd_utils import DefaultBoxes


class SSDLoss(_Loss):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels

    L = (2 - alpha) * L_l1 + alpha * L_cls, where
        * L_cls = [CE of all positives] + [CE of the hardest backgrounds]
          and the second term is built from [3 * positive pairs] background cells with the highest CE
        * L_l1 = [SmoothL1Loss for all positives]
    """

    def __init__(self, dboxes: DefaultBoxes, alpha: float = 1.0, iou_thresh: float = 0.5, neg_pos_ratio: float = 3.):
        super(SSDLoss, self).__init__()
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh
        self.alpha = alpha
        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False)

        self.con_loss = nn.CrossEntropyLoss(reduce=False)
        self.iou_thresh = iou_thresh
        self.neg_pos_ratio = neg_pos_ratio

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
        each destination bbox is matched to ground truth with the highest IoU with it, with a condition that
        the IoU is above self.iou_thresh.
        A GT bboxes can be duplicated to a few destination boxes, but all GT boxes aren't guaranteed to be used.

        :param targets: a tensor containing the boxes for a single image;
                        shape [num_boxes, 6] (image_id, label, x, y, w, h)
        :return:        two tensors
                        boxes - shape of dboxes [4, num_dboxes] (x,y,w,h)
                        labels - sahpe [num_dboxes]
        """
        target_locations = self.dboxes.data.clone().squeeze()
        target_labels = torch.zeros((self.dboxes.data.shape[2])).to(self.dboxes.device)

        if len(targets) > 0:
            boxes = targets[:, 2:]
            ious = calculate_bbox_iou_matrix(boxes, self.dboxes.data.squeeze().T, x1y1x2y2=False)

            # one best GT > self.iou_thresh for EACH cell,
            # but it's a problem that not all GTs are guaranteed to get a cell pair
            values, indices = torch.max(ious, dim=0)
            mask = values > self.iou_thresh

            target_locations[:, mask] = targets[indices[mask], 2:].T
            target_labels[mask] = targets[indices[mask], 1] + 1

        return target_locations, target_labels

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

        mask = batch_target_labels > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._norm_relative_bbox(batch_target_locations)

        # SUM ON FOUR COORDINATES, AND MASK
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float() * sl1).sum(dim=1)

        # HARD NEGATIVE MINING
        con = self.con_loss(plabel, batch_target_labels)

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

        # AVOID NO OBJECT DETECTED
        total_loss = (2 - self.alpha) * sl1 + self.alpha * closs
        num_mask = (pos_num > 0).float()  # a mask with 0 for images that have no positive pairs at all
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # normalize by the number of positive pairs

        return ret, torch.cat((sl1.mean().unsqueeze(0), closs.mean().unsqueeze(0), ret.unsqueeze(0))).detach()
