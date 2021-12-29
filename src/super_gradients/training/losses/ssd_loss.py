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
    """

    def __init__(self, dboxes: DefaultBoxes, alpha: float = 1.0):
        super(SSDLoss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh
        self.alpha = alpha
        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False)

        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def _norm_relative_bbox(self, loc):
        """
        convert bbox locations into relative locations (relative to the dboxes) and normalized by w,h
        :param loc a tensor of shape [batch, 4, num_boxes]
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def match_dboxes(self, targets):
        """
        convert ground truth boxes into a tensor with the same size as dboxes. each gt bbox is matched to every
        destination box which overlaps it over 0.5 (IoU). so some gt bboxes can be duplicated to a few destination boxes
        :param targets: a tensor containing the boxes for a single image. shape [num_boxes, 5] (x,y,w,h,label)
        :return: two tensors
            boxes - shape of dboxes [4, num_dboxes] (x,y,w,h)
            labels - sahpe [num_dboxes]
        """
        target_locations = self.dboxes.data.clone().squeeze()
        target_labels = torch.zeros((self.dboxes.data.shape[2])).to(self.dboxes.device)

        if len(targets) > 0:
            boxes = targets[:, 2:]
            ious = calculate_bbox_iou_matrix(boxes, self.dboxes.data.squeeze().T, x1y1x2y2=False)

            values, indices = torch.max(ious, dim=0)
            mask = values > 0.5

            target_locations[:, mask] = targets[indices[mask], 2:].T
            target_labels[mask] = targets[indices[mask], 1]

        return target_locations, target_labels

    def forward(self, predictions, targets):
        """
        Compute the loss
            :param predictions - predictions tensor coming from the network. shape [N, num_classes+4, num_dboxes]
            were the first four items are (x,y,w,h) and the rest are class confidence
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

        # POSITIVE MASK WILL NEVER SELECTED
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # NUMBER OF NEGATIVE THREE TIMES POSITIVE
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        # AVOID NO OBJECT DETECTED
        total_loss = (2 - self.alpha) * sl1 + self.alpha * closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)

        return ret, torch.cat((sl1.mean().unsqueeze(0), closs.mean().unsqueeze(0), ret.unsqueeze(0))).detach()
