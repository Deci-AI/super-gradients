"""
Based on https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0 license)

"""

import logging
from typing import List, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils import torch_version_is_greater_or_equal
from super_gradients.training.utils.detection_utils import calculate_bbox_iou_matrix

logger = get_logger(__name__)


class IOUloss(nn.Module):
    """
    IoU loss with the following supported loss types:
    :param reduction: One of ["mean", "sum", "none"] reduction to apply to the computed loss (Default="none")
    :param loss_type: One of ["iou", "giou"] where:
            * 'iou' for
                (1 - iou^2)
            * 'giou' according to "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
                (1 - giou), where giou = iou - (cover_box - union_box)/cover_box
    """

    def __init__(self, reduction: str = "none", loss_type: str = "iou"):
        super(IOUloss, self).__init__()
        self._validate_args(loss_type, reduction)
        self.reduction = reduction
        self.loss_type = loss_type

    @staticmethod
    def _validate_args(loss_type, reduction):
        supported_losses = ["iou", "giou"]
        supported_reductions = ["mean", "sum", "none"]
        if loss_type not in supported_losses:
            raise ValueError("Illegal loss_type value: " + loss_type + ", expected one of: " + str(supported_losses))
        if reduction not in supported_reductions:
            raise ValueError("Illegal reduction value: " + reduction + ", expected one of: " + str(supported_reductions))

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


@register_loss(Losses.YOLOX_LOSS)
class YoloXDetectionLoss(_Loss):
    """
    Calculate YOLOX loss:
    L = L_objectivness + L_iou + L_classification + 1[use_l1]*L_l1

    where:
        * L_iou, L_classification and L_l1 are calculated only between cells and targets that suit them;
        * L_objectivness is calculated for all cells.

        L_classification:
            for cells that have suitable ground truths in their grid locations add BCEs
            to force a prediction of IoU with a GT in a multi-label way
            Coef: 1.
        L_iou:
            for cells that have suitable ground truths in their grid locations
            add (1 - IoU^2), IoU between a predicted box and each GT box, force maximum IoU
            Coef: 5.
        L_l1:
            for cells that have suitable ground truths in their grid locations
            l1 distance between the logits and GTs in “logits” format (the inverse of “logits to predictions” ops)
            Coef: 1[use_l1]
        L_objectness:
            for each cell add BCE with a label of 1 if there is GT assigned to the cell
            Coef: 1

    :param strides:                 List of Yolo levels output grid sizes (i.e [8, 16, 32]).
    :param num_classes:             Number of classes.
    :param use_l1:                  Controls the L_l1 Coef as discussed above (default=False).
    :param center_sampling_radius:  Sampling radius used for center sampling when creating the fg mask (default=2.5).
    :param iou_type:                Iou loss type, one of ["iou","giou"] (deafult="iou").
    """

    def __init__(self, strides: list, num_classes: int, use_l1: bool = False, center_sampling_radius: float = 2.5, iou_type: str = "iou"):
        super().__init__()
        self.grids = [torch.zeros(1)] * len(strides)
        self.strides = strides
        self.num_classes = num_classes

        self.center_sampling_radius = center_sampling_radius
        self.use_l1 = use_l1
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none", loss_type=iou_type)

    @property
    def component_names(self) -> List[str]:
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return ["iou", "obj", "cls", "l1", "num_fg", "Loss"]

    def forward(self, model_output: Union[list, Tuple[torch.Tensor, List]], targets: torch.Tensor):
        """
        :param model_output: Union[list, Tuple[torch.Tensor, List]]:
             When list-
              output from all Yolo levels, each of shape [Batch x 1 x GridSizeY x GridSizeX x (4 + 1 + Num_classes)]
             And when tuple- the second item is the described list (first item is discarded)

        :param targets: torch.Tensor: Num_targets x (4 + 2)], values on dim 1 are: image id in a batch, class, box x y w h

        :return: loss, all losses separately in a detached tensor
        """
        if isinstance(model_output, tuple) and len(model_output) == 2:
            # in test/eval mode the Yolo model outputs a tuple where the second item is the raw predictions
            _, predictions = model_output
        else:
            predictions = model_output

        return self._compute_loss(predictions, targets)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """
        Creates a tensor of xy coordinates of size (1,1,nx,ny,2)

        :param nx: int: cells along x axis (default=20)
        :param ny: int: cells along the y axis (default=20)
        :return: torch.tensor of xy coordinates of size (1,1,nx,ny,2)
        """
        if torch_version_is_greater_or_equal(1, 10):
            # https://github.com/pytorch/pytorch/issues/50276
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        else:
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _compute_loss(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param predictions:     output from all Yolo levels, each of shape
                                [Batch x 1 x GridSizeY x GridSizeX x (4 + 1 + Num_classes)]
        :param targets:         [Num_targets x (4 + 2)], values on dim 1 are: image id in a batch, class, box x y w h

        :return:                loss, all losses separately in a detached tensor
        """
        x_shifts, y_shifts, expanded_strides, transformed_outputs, raw_outputs = self.prepare_predictions(predictions)

        bbox_preds = transformed_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = transformed_outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = transformed_outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        total_num_anchors = transformed_outputs.shape[1]
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg, num_gts = 0.0, 0.0

        for image_idx in range(transformed_outputs.shape[0]):
            labels_im = targets[targets[:, 0] == image_idx]
            num_gt = labels_im.shape[0]
            num_gts += num_gt
            if num_gt == 0:
                cls_target = transformed_outputs.new_zeros((0, self.num_classes))
                reg_target = transformed_outputs.new_zeros((0, 4))
                l1_target = transformed_outputs.new_zeros((0, 4))
                obj_target = transformed_outputs.new_zeros((total_num_anchors, 1))
                fg_mask = transformed_outputs.new_zeros(total_num_anchors).bool()
            else:
                # GT boxes to image coordinates
                gt_bboxes_per_image = labels_im[:, 2:6].clone()
                gt_classes = labels_im[:, 1]
                bboxes_preds_per_image = bbox_preds[image_idx]

                try:
                    # assign cells to ground truths, at most one GT per cell
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                        image_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )

                # TODO: CHECK IF ERROR IS CUDA OUT OF MEMORY
                except RuntimeError:
                    logging.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                                   CPU mode is applied in this batch. If you want to avoid this issue, \
                                   try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                        image_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        transformed_outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            # collect targets for all loss terms over the whole batch
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(transformed_outputs.dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        # concat all targets over the batch (get rid of batch dim)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        # loss terms divided by the total number of foregrounds
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets).sum() / num_fg
        loss_obj = self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets).sum() / num_fg
        loss_cls = self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets).sum() / num_fg
        if self.use_l1:
            loss_l1 = self.l1_loss(raw_outputs.view(-1, 4)[fg_masks], l1_targets).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            torch.cat(
                (
                    loss_iou.unsqueeze(0),
                    loss_obj.unsqueeze(0),
                    loss_cls.unsqueeze(0),
                    torch.tensor(loss_l1).unsqueeze(0).to(loss.device),
                    torch.tensor(num_fg / max(num_gts, 1)).unsqueeze(0).to(loss.device),
                    loss.unsqueeze(0),
                )
            ).detach(),
        )

    def prepare_predictions(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw outputs of the network into a format that merges outputs from all levels
        :param predictions:     output from all Yolo levels, each of shape
                                [Batch x 1 x GridSizeY x GridSizeX x (4 + 1 + Num_classes)]
        :return:    5 tensors representing predictions:
                        * x_shifts: shape [1 x * num_cells x 1],
                          where num_cells = grid1X * grid1Y + grid2X * grid2Y + grid3X * grid3Y,
                          x coordinate on the grid cell the prediction is coming from
                        * y_shifts: shape [1 x  num_cells x 1],
                          y coordinate on the grid cell the prediction is coming from
                        * expanded_strides: shape [1 x num_cells x 1],
                          stride of the output grid the prediction is coming from
                        * transformed_outputs: shape [batch_size x num_cells x (num_classes + 5)],
                          predictions with boxes in real coordinates and logprobabilities
                        * raw_outputs: shape [batch_size x num_cells x (num_classes + 5)],
                          raw predictions with boxes and confidences as logits

        """
        raw_outputs = []
        transformed_outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, output in enumerate(predictions):
            batch_size, num_anchors, h, w, num_outputs = output.shape

            # IN FIRST PASS CREATE GRIDS ACCORDING TO OUTPUT SHAPE (BATCH,1,IMAGE_H/STRIDE,IMAGE_2/STRIDE,NUM_CLASSES+5)
            if self.grids[k].shape[2:4] != output.shape[2:4]:
                self.grids[k] = self._make_grid(w, h).type_as(output)

            # e.g. [batch_size, 1, 28, 28, 85] -> [batch_size, 784, 85]
            output_raveled = output.reshape(batch_size, num_anchors * h * w, num_outputs)
            # e.g [1, 784, 2]
            grid_raveled = self.grids[k].view(1, num_anchors * h * w, 2)
            if self.use_l1:
                # e.g [1, 784, 4]
                raw_outputs.append(output_raveled[:, :, :4].clone())

            # box logits to coordinates
            centers = (output_raveled[..., :2] + grid_raveled) * self.strides[k]
            wh = torch.exp(output_raveled[..., 2:4]) * self.strides[k]
            classes = output_raveled[..., 4:]
            output_raveled = torch.cat([centers, wh, classes], -1)

            # outputs with boxes in real coordinates, probs as logits
            transformed_outputs.append(output_raveled)
            # x cell coordinates of all 784 predictions, 0, 0, 0, ..., 1, 1, 1, ...
            x_shifts.append(grid_raveled[:, :, 0])
            # y cell coordinates of all 784 predictions, 0, 1, 2, ..., 0, 1, 2, ...
            y_shifts.append(grid_raveled[:, :, 1])
            # e.g. [1, 784, stride of this level (one of [8, 16, 32])]
            expanded_strides.append(torch.zeros(1, grid_raveled.shape[1]).fill_(self.strides[k]).type_as(output))

        # all 4 below have shapes of [batch_size , num_cells, num_values_pre_cell]
        # where num_anchors * num_cells is e.g. 1 * (28 * 28 + 14 * 14 + 17 * 17)
        transformed_outputs = torch.cat(transformed_outputs, 1)
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            raw_outputs = torch.cat(raw_outputs, 1)

        return x_shifts, y_shifts, expanded_strides, transformed_outputs, raw_outputs

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        """
        :param l1_target:   tensor of zeros of shape [Num_cell_gt_pairs x 4]
        :param gt:          targets in coordinates [Num_cell_gt_pairs x (4 + 1 + num_classes)]

        :return:            targets in the format corresponding to logits
        """
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        image_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
        ious_loss_cost_coeff=3.0,
        outside_boxes_and_center_cost_coeff=100000.0,
    ):
        """
        Match cells to ground truth:
            * at most 1 GT per cell
            * dynamic number of cells per GT

        :param outside_boxes_and_center_cost_coeff: float: Cost coefficiant of cells the radius and bbox of gts in dynamic
         matching (default=100000).
        :param ious_loss_cost_coeff: float: Cost coefficiant for iou loss in dynamic matching (default=3).
        :param image_idx: int: Image index in batch.
        :param num_gt: int: Number of ground trunth targets in the image.
        :param total_num_anchors: int: Total number of possible bboxes = sum of all grid cells.
        :param gt_bboxes_per_image: torch.Tensor: Tensor of gt bboxes for  the image, shape: (num_gt, 4).
        :param gt_classes: torch.Tesnor: Tensor of the classes in the image, shape: (num_preds,4).
        :param bboxes_preds_per_image: Tensor of the classes in the image, shape: (num_preds).
        :param expanded_strides: torch.Tensor: Stride of the output grid the prediction is coming from,
            shape (1 x num_cells x 1).
        :param x_shifts: torch.Tensor: X's in cell coordinates, shape (1,num_cells,1).
        :param y_shifts: torch.Tensor: Y's in cell coordinates, shape (1,num_cells,1).
        :param cls_preds: torch.Tensor: Class predictions in all cells, shape (batch_size, num_cells).
        :param obj_preds: torch.Tensor: Objectness predictions in all cells, shape (batch_size, num_cells).
        :param mode: str: One of ["gpu","cpu"], Controls the device the assignment operation should be taken place on (deafult="gpu")

        """
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # create a mask for foreground cells
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[image_idx][fg_mask]
        obj_preds_ = obj_preds[image_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # calculate cost between all foregrounds and all ground truths (used only for matching)
        pair_wise_ious = calculate_bbox_iou_matrix(gt_bboxes_per_image, bboxes_preds_per_image, x1y1x2y2=False)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes)
        gt_cls_per_image = gt_cls_per_image.float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_

        cost = pair_wise_cls_loss + ious_loss_cost_coeff * pair_wise_ious_loss + outside_boxes_and_center_cost_coeff * (~is_in_boxes_and_center)

        # further filter foregrounds: create pairs between cells and ground truth, based on cost and IoUs
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        # discard tensors related to cost
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        """
        Create a mask for all cells, mask in only foreground: cells that have a center located:
            * withing a GT box;
            OR
            * within a fixed radius around a GT box (center sampling);

        :param num_gt: int: Number of ground trunth targets in the image.
        :param total_num_anchors: int: Sum of all grid cells.
        :param gt_bboxes_per_image: torch.Tensor: Tensor of gt bboxes for  the image, shape: (num_gt, 4).
        :param expanded_strides: torch.Tensor: Stride of the output grid the prediction is coming from,
            shape (1 x num_cells x 1).
        :param x_shifts: torch.Tensor: X's in cell coordinates, shape (1,num_cells,1).
        :param y_shifts: torch.Tensor: Y's in cell coordinates, shape (1,num_cells,1).

        :return is_in_boxes_anchor, is_in_boxes_and_center
            where:
             - is_in_boxes_anchor masks the cells that their cell center is  inside a gt bbox and within
                self.center_sampling_radius cells away, without reduction (i.e shape=(num_gts, num_fgs))
             - is_in_boxes_and_center masks the cells that their center is either inside a gt bbox or within
                self.center_sampling_radius cells away, shape (num_fgs)
        """

        expanded_strides_per_image = expanded_strides[0]

        # cell coordinates, shape [n_predictions] -> repeated to [n_gts, n_predictions]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = (y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        # FIND CELL CENTERS THAT ARE WITHIN GROUND TRUTH BOXES

        # ground truth boxes, shape [n_gts] -> repeated to [n_gts, n_predictions]
        # from (c1, c2, w, h) to left, right, top, bottom
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)

        # check which cell centers lay within the ground truth boxes
        b_l = x_centers_per_image - gt_bboxes_per_image_l  # x - l > 0 when l is on the lest from x
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # shape [n_gts, n_predictions]

        # to claim that a cell center is inside a gt box all 4 differences calculated above should be positive
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # shape [n_gts, n_predictions]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0  # shape [n_predictions], whether a cell is inside at least one gt

        # FIND CELL CENTERS THAT ARE WITHIN +- self.center_sampling_radius CELLS FROM GROUND TRUTH BOXES CENTERS

        # define fake boxes: instead of ground truth boxes step +- self.center_sampling_radius from their centers
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - self.center_sampling_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + self.center_sampling_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - self.center_sampling_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + self.center_sampling_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes OR in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        # in boxes AND in centers, preserving a shape [num_GTs x num_FGs]
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        :param cost:            pairwise cost, [num_FGs x num_GTs]
        :param pair_wise_ious:  pairwise IoUs, [num_FGs x num_GTs]
        :param gt_classes:      class of each GT
        :param num_gt:          number of GTs

        :return num_fg, (number of foregrounds)
                gt_matched_classes, (the classes that have been matched with fgs)
                pred_ious_this_matching
                matched_gt_inds
        """
        # create a matrix with shape [num_GTs x num_FGs]
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        # for each GT get a dynamic k of foregrounds with a minimum cost: k = int(sum[top 10 IoUs])
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            try:
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            except Exception:
                logger.warning("cost[gt_idx]: " + str(cost[gt_idx]) + " dynamic_ks[gt_idx]L " + str(dynamic_ks[gt_idx]))
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        # leave at most one GT per foreground, chose the one with the smallest cost
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


@register_loss(Losses.YOLOX_FAST_LOSS)
class YoloXFastDetectionLoss(YoloXDetectionLoss):
    """
    A completely new implementation of YOLOX loss.
    This is NOT an equivalent implementation to the regular yolox loss.

    * Completely avoids using loops compared to the nested loops in the original implementation.
        As a result runs much faster (speedup depends on the type of GPUs, their count, the batch size, etc.).
    * Tensors format is very different the original implementation.
        Tensors contain image ids, ground truth ids and anchor ids as values to support variable length data.
    * There are differences in terms of the algorithm itself:
    1. When computing a dynamic k for a ground truth,
        in the original implementation they consider the sum of top 10 predictions sorted by ious among the initial
        foregrounds of any ground truth in the image,
        while in our implementation we consider only the initial foreground of that particular ground truth.
        To compensate for that difference we introduce the dynamic_ks_bias hyperparamter which makes the dynamic ks larger.
    2. When computing the k matched detections for a ground truth,
        in the original implementation they consider the initial foregrounds of any ground truth in the image as candidates,
        while in our implementation we consider only the initial foreground of that particular ground truth as candidates.
        We believe that this difference is minor.

    :param dynamic_ks_bias: hyperparameter to compensate for the discrepancies between the regular loss and this loss.
    :param sync_num_fgs:    sync num of fgs.
                            Can be used for DDP training.
    :param obj_loss_fix:    devide by total of num anchors instead num of matching fgs.
                            Can be used for objectness loss.
    """

    def __init__(
        self, strides, num_classes, use_l1=False, center_sampling_radius=2.5, iou_type="iou", dynamic_ks_bias=1.1, sync_num_fgs=False, obj_loss_fix=False
    ):
        super().__init__(strides=strides, num_classes=num_classes, use_l1=use_l1, center_sampling_radius=center_sampling_radius, iou_type=iou_type)

        self.dynamic_ks_bias = dynamic_ks_bias
        self.sync_num_fgs = sync_num_fgs
        self.obj_loss_fix = obj_loss_fix

    def _compute_loss(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        L = L_objectness + L_iou + L_classification + 1[no_aug_epoch]*L_l1
        where:
            * L_iou, L_classification and L_l1 are calculated only between cells and targets that suit them;
            * L_objectness is calculated for all cells.

        L_classification:
            for cells that have suitable ground truths in their grid locations add BCEs
            to force a prediction of IoU with a GT in a multi-label way
            Coef: 1.
        L_iou:
            for cells that have suitable ground truths in their grid locations
            add (1 - IoU^2), IoU between a predicted box and each GT box, force maximum IoU
            Coef: 1.
        L_l1:
            for cells that have suitable ground truths in their grid locations
            l1 distance between the logits and GTs in “logits” format (the inverse of “logits to predictions” ops)
            Coef: 1[no_aug_epoch]
        L_objectness:
            for each cell add BCE with a label of 1 if there is GT assigned to the cell
            Coef: 5

        :param predictions:     output from all Yolo levels, each of shape
                                [Batch x Num_Anchors x GridSizeY x GridSizeX x (4 + 1 + Num_classes)]
        :param targets:         [Num_targets x (4 + 2)], values on dim 1 are: image id in a batch, class, box x y w h

        :return:                loss, all losses separately in a detached tensor
        """
        x_shifts, y_shifts, expanded_strides, transformed_outputs, raw_outputs = self.prepare_predictions(predictions)

        bbox_preds = transformed_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = transformed_outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = transformed_outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # assign cells to ground truths, at most one GT per cell
        matched_fg_ids, matched_gt_classes, matched_gt_ids, matched_img_ids, matched_ious, flattened_gts = self._compute_matching(
            bbox_preds, cls_preds, obj_preds, expanded_strides, x_shifts, y_shifts, targets
        )

        num_gts = flattened_gts.shape[0]
        num_gts_clamped = max(flattened_gts.shape[0], 1)
        num_fg = max(matched_gt_ids.shape[0], 1)
        total_num_anchors = max(transformed_outputs.shape[0] * transformed_outputs.shape[1], 1)

        cls_targets = F.one_hot(matched_gt_classes.to(torch.int64), self.num_classes) * matched_ious.unsqueeze(dim=1)
        obj_targets = transformed_outputs.new_zeros((transformed_outputs.shape[0], transformed_outputs.shape[1]))
        obj_targets[matched_img_ids, matched_fg_ids] = 1
        reg_targets = flattened_gts[matched_gt_ids][:, 1:]
        if self.use_l1 and num_gts > 0:
            l1_targets = self.get_l1_target(
                transformed_outputs.new_zeros((num_fg, 4)),
                flattened_gts[matched_gt_ids][:, 1:],
                expanded_strides.squeeze()[matched_fg_ids],
                x_shifts=x_shifts.squeeze()[matched_fg_ids],
                y_shifts=y_shifts.squeeze()[matched_fg_ids],
            )
        if self.sync_num_fgs and dist.group.WORLD is not None:
            num_fg = torch.scalar_tensor(num_fg).to(matched_gt_ids.device)
            dist.all_reduce(num_fg, op=torch._C._distributed_c10d.ReduceOp.AVG)

        loss_iou = self.iou_loss(bbox_preds[matched_img_ids, matched_fg_ids], reg_targets).sum() / num_fg
        loss_obj = self.bcewithlog_loss(obj_preds.squeeze(-1), obj_targets).sum() / (total_num_anchors if self.obj_loss_fix else num_fg)
        loss_cls = self.bcewithlog_loss(cls_preds[matched_img_ids, matched_fg_ids], cls_targets).sum() / num_fg

        if self.use_l1 and num_gts > 0:
            loss_l1 = self.l1_loss(raw_outputs[matched_img_ids, matched_fg_ids], l1_targets).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            torch.cat(
                (
                    loss_iou.unsqueeze(0),
                    loss_obj.unsqueeze(0),
                    loss_cls.unsqueeze(0),
                    torch.tensor(loss_l1).unsqueeze(0).to(transformed_outputs.device),
                    torch.tensor(num_fg / num_gts_clamped).unsqueeze(0).to(transformed_outputs.device),
                    loss.unsqueeze(0),
                )
            ).detach(),
        )

    def _get_initial_matching(
        self, gt_bboxes: torch.Tensor, expanded_strides: torch.Tensor, x_shifts: torch.Tensor, y_shifts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get candidates using a mask for all cells.
        Mask in only foreground cells that have a center located:
            * withing a GT box (param: is_in_boxes);
            OR
            * within a fixed radius around a GT box (center sampling) (param: is_in_centers);

        return:
            initial_matching: get a list of candidates pairs of (gt box id, anchor box id) based on cell = is_in_boxes | is_in_centers.
                              shape: [num_candidates, 2]
            strong candidate mask: get a list whether a candidate is a strong one or not.
                                   strong candidate is a cell from is_in_boxes & is_in_centers.
                                   shape: [num_candidates].
        """
        cell_x_centers = (x_shifts + 0.5) * expanded_strides
        cell_y_centers = (y_shifts + 0.5) * expanded_strides

        gt_bboxes_x_centers = gt_bboxes[:, 0].unsqueeze(1)
        gt_bboxes_y_centers = gt_bboxes[:, 1].unsqueeze(1)

        gt_bboxes_half_w = (0.5 * gt_bboxes[:, 2]).unsqueeze(1)
        gt_bboxes_half_h = (0.5 * gt_bboxes[:, 3]).unsqueeze(1)

        is_in_boxes = (
            (cell_x_centers > gt_bboxes_x_centers - gt_bboxes_half_w)
            & (gt_bboxes_x_centers + gt_bboxes_half_w > cell_x_centers)
            & (cell_y_centers > gt_bboxes_y_centers - gt_bboxes_half_h)
            & (gt_bboxes_y_centers + gt_bboxes_half_h > cell_y_centers)
        )

        radius_shifts = 2.5 * expanded_strides

        is_in_centers = (
            (cell_x_centers + radius_shifts > gt_bboxes_x_centers)
            & (gt_bboxes_x_centers > cell_x_centers - radius_shifts)
            & (cell_y_centers + radius_shifts > gt_bboxes_y_centers)
            & (gt_bboxes_y_centers > cell_y_centers - radius_shifts)
        )

        initial_mask = is_in_boxes | is_in_centers
        initial_matching = initial_mask.nonzero()
        strong_candidate_mask = (is_in_boxes & is_in_centers)[initial_mask]

        return initial_matching[:, 0], initial_matching[:, 1], strong_candidate_mask

    @torch.no_grad()
    def _compute_matching(
        self,
        bbox_preds: torch.Tensor,
        cls_preds: torch.Tensor,
        obj_preds: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        labels: torch.Tensor,
        ious_loss_cost_coeff: float = 3.0,
        outside_boxes_and_center_cost_coeff: float = 100000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match cells to ground truth:
            * at most 1 GT per cell
            * dynamic number of cells per GT

        :param bbox_preds: predictions of bounding boxes. shape [batch, n_anchors_all, 4]
        :param cls_preds:  predictions of class.          shape [batch, n_anchors_all, n_cls]
        :param obj_preds:  predictions for objectness.    shape [batch, n_anchors_all, 1]
        :param expanded_strides:  stride of the output grid the prediction is coming from. shape [1, n_anchors_all]
        :param x_shifts: x coordinate on the grid cell the prediction is coming from.      shape [1, n_anchors_all]
        :param y_shifts: y coordinate on the grid cell the prediction is coming from.      shape [1, n_anchors_all]
        :param labels:   labels for each grid cell.  shape [n_anchors_all, (4 + 2)]
        :return: candidate_fg_ids       shape [num_fg]
                 candidate_gt_classes   shape [num_fg]
                 candidate_gt_ids       shape [num_fg]
                 candidate_img_ids      shape [num_fg]
                 candidate_ious         shape [num_fg]
                 flattened_gts          shape [num_gts, 5]
        """

        flattened_gts, gt_id_to_img_id = labels[:, 1:], labels[:, 0].type(torch.int64)

        # COMPUTE CANDIDATES
        candidate_gt_ids, candidate_fg_ids, strong_candidate_mask = self._get_initial_matching(flattened_gts[:, 1:], expanded_strides, x_shifts, y_shifts)
        candidate_img_ids = gt_id_to_img_id[candidate_gt_ids]
        candidate_gts_bbox = flattened_gts[candidate_gt_ids, 1:]
        candidate_det_bbox = bbox_preds[candidate_img_ids, candidate_fg_ids]

        # COMPUTE DYNAMIC KS
        candidate_ious = self._calculate_pairwise_bbox_iou(candidate_gts_bbox, candidate_det_bbox, xyxy=False)
        dynamic_ks, matching_index_to_dynamic_k_index = self._compute_dynamic_ks(candidate_gt_ids, candidate_ious, self.dynamic_ks_bias)
        del candidate_gts_bbox, candidate_det_bbox

        # ORDER CANDIDATES BY COST
        candidate_gt_classes = flattened_gts[candidate_gt_ids, 0]
        cost_order = self._compute_cost_order(
            self.num_classes,
            candidate_img_ids,
            candidate_gt_classes,
            candidate_fg_ids,
            candidate_ious,
            cls_preds,
            obj_preds,
            strong_candidate_mask,
            ious_loss_cost_coeff,
            outside_boxes_and_center_cost_coeff,
        )

        candidate_gt_ids = candidate_gt_ids[cost_order]
        candidate_gt_classes = candidate_gt_classes[cost_order]
        candidate_img_ids = candidate_img_ids[cost_order]
        candidate_fg_ids = candidate_fg_ids[cost_order]
        candidate_ious = candidate_ious[cost_order]
        matching_index_to_dynamic_k_index = matching_index_to_dynamic_k_index[cost_order]
        del cost_order

        # FILTER MATCHING TO LOWEST K COST MATCHES PER GT
        ranks = self._compute_ranks(candidate_gt_ids)
        corresponding_dynamic_ks = dynamic_ks[matching_index_to_dynamic_k_index]
        topk_mask = ranks < corresponding_dynamic_ks

        candidate_gt_ids = candidate_gt_ids[topk_mask]
        candidate_gt_classes = candidate_gt_classes[topk_mask]
        candidate_img_ids = candidate_img_ids[topk_mask]
        candidate_fg_ids = candidate_fg_ids[topk_mask]
        candidate_ious = candidate_ious[topk_mask]
        del ranks, topk_mask, dynamic_ks, matching_index_to_dynamic_k_index, corresponding_dynamic_ks

        # FILTER MATCHING TO AT MOST 1 MATCH FOR DET BY TAKING THE LOWEST COST MATCH
        candidate_img_and_fg_ids_combined = self._combine_candidates_img_id_fg_id(candidate_img_ids, candidate_fg_ids)
        top1_mask = self._compute_is_first_mask(candidate_img_and_fg_ids_combined)
        candidate_gt_ids = candidate_gt_ids[top1_mask]
        candidate_gt_classes = candidate_gt_classes[top1_mask]
        candidate_fg_ids = candidate_fg_ids[top1_mask]
        candidate_img_ids = candidate_img_ids[top1_mask]
        candidate_ious = candidate_ious[top1_mask]

        return candidate_fg_ids, candidate_gt_classes, candidate_gt_ids, candidate_img_ids, candidate_ious, flattened_gts

    def _combine_candidates_img_id_fg_id(self, candidate_img_ids, candidate_anchor_ids):
        """
        Create one dim tensor with unique pairs of img_id and fg_id.
        e.g: candidate_img_ids = [0,1,0,0]
             candidate_fg_ids = [0,0,0,1]
             result = [0,1,0,2]
        """
        candidate_img_and_fg_ids_combined = torch.stack((candidate_img_ids, candidate_anchor_ids), dim=1).unique(dim=0, return_inverse=True)[1]
        return candidate_img_and_fg_ids_combined

    def _compute_dynamic_ks(self, ids: torch.Tensor, ious: torch.Tensor, dynamic_ks_bias) -> torch.Tensor:
        """
        :param ids:                 ids of GTs, shape: [num_candidates]
        :param ious:                pairwise IoUs, shape: [num_candidates]
        :param dynamic_ks_bias:     multiply the resulted k to compensate the regular loss
        """
        assert len(ids.shape) == 1, "ids must be of shape [num_candidates]"
        assert len(ious.shape) == 1, "ious must be of shape [num_candidates]"
        assert ids.shape[0] == ious.shape[0], "num of ids.shape[0] must be the same as num of ious.shape[0]"
        # sort ious and ids by ious
        ious, ious_argsort = ious.sort(descending=True)
        ids = ids[ious_argsort]

        # stable sort indices, so that ious are first sorted by id and second by value
        ids, ids_argsort = ids.sort(stable=True)
        ious = ious[ids_argsort]

        unique_ids, ids_index_to_unique_ids_index = ids.unique_consecutive(dim=0, return_inverse=True)
        num_unique_ids = unique_ids.shape[0]

        if ids.shape[0] > 10:
            is_in_top_10 = torch.cat((torch.ones((10,), dtype=torch.bool, device=ids.device), ids[10:] != ids[:-10]))
        else:
            is_in_top_10 = torch.ones_like(ids, dtype=torch.bool)

        dynamic_ks = torch.zeros((num_unique_ids,), dtype=ious.dtype, device=ious.device)
        dynamic_ks.index_put_((ids_index_to_unique_ids_index,), is_in_top_10 * ious, accumulate=True)
        if dynamic_ks_bias is not None:
            dynamic_ks *= dynamic_ks_bias
        dynamic_ks = dynamic_ks.long().clamp(min=1)

        all_argsort = ious_argsort[ids_argsort]
        inverse_all_argsort = torch.zeros_like(ious_argsort)
        inverse_all_argsort[all_argsort] = torch.arange(all_argsort.shape[0], dtype=all_argsort.dtype, device=all_argsort.device)

        return dynamic_ks, ids_index_to_unique_ids_index[inverse_all_argsort]

    def _compute_cost_order(
        self,
        num_classes,
        candidate_gt_img_ids: torch.Tensor,
        candidate_gt_classes: torch.Tensor,
        candidate_anchor_ids: torch.Tensor,
        candidate_ious: torch.Tensor,
        cls_preds: torch.Tensor,
        obj_preds: torch.Tensor,
        strong_candidate_mask: torch.Tensor,
        ious_loss_cost_coeff: float,
        outside_boxes_and_center_cost_coeff: float,
    ) -> torch.Tensor:
        gt_cls_per_image = F.one_hot(candidate_gt_classes.to(torch.int64), num_classes).float()
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds[candidate_gt_img_ids, candidate_anchor_ids].float().sigmoid_()
                * obj_preds[candidate_gt_img_ids, candidate_anchor_ids].float().sigmoid_()
            )
            pair_wise_cls_cost = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)

        ious_cost = -torch.log(candidate_ious + 1e-8)
        cost = pair_wise_cls_cost + ious_loss_cost_coeff * ious_cost + outside_boxes_and_center_cost_coeff * strong_candidate_mask.logical_not()
        return cost.argsort()

    def _calculate_pairwise_bbox_iou(self, bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, xyxy=True) -> torch.Tensor:
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, :2] - bboxes_a[:, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, :2] + bboxes_a[:, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        return area_i / (area_a + area_b - area_i)

    def _compute_ranks(self, ids: torch.Tensor) -> torch.Tensor:
        ids, ids_argsort = ids.sort(stable=True)

        if ids.shape[0] > 1:
            is_not_first = torch.cat((torch.zeros((1,), dtype=torch.bool, device=ids.device), ids[1:] == ids[:-1]))
        else:
            is_not_first = torch.zeros_like(ids, dtype=torch.bool)

        subtract = torch.arange(ids.shape[0], dtype=ids_argsort.dtype, device=ids.device)
        subtract[is_not_first] = 0
        subtract = subtract.cummax(dim=0)[0]
        rank = torch.arange(ids.shape[0], dtype=ids_argsort.dtype, device=ids.device) - subtract

        inverse_argsort = torch.zeros_like(ids_argsort)
        inverse_argsort[ids_argsort] = torch.arange(ids_argsort.shape[0], dtype=ids_argsort.dtype, device=ids_argsort.device)

        return rank[inverse_argsort]

    def _compute_is_first_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Filter fg that matches two gts.
        """
        ids, ids_argsort = ids.sort(stable=True)

        if ids.shape[0] > 1:
            is_first = torch.cat((torch.ones((1,), dtype=torch.bool, device=ids.device), ids[1:] != ids[:-1]))
        else:
            is_first = torch.ones_like(ids, dtype=torch.bool)

        inverse_argsort = torch.zeros_like(ids_argsort)
        inverse_argsort[ids_argsort] = torch.arange(ids_argsort.shape[0], dtype=ids_argsort.dtype, device=ids_argsort.device)

        return is_first[inverse_argsort]
