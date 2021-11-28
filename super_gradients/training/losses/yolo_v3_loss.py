import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from super_gradients.training.utils.detection_utils import build_detection_targets, calculate_bbox_iou_elementwise


class YoLoV3DetectionLoss(_Loss):
    """
    YoLoV3DetectionLoss - Loss Class for Object Detection
    """

    def __init__(self, model: nn.Module, cls_pw: float = 1., obj_pw: float = 1., giou: float = 3.54, obj: float = 64.3,
                 cls: float = 37.4):
        super(YoLoV3DetectionLoss, self).__init__()
        self.model = model
        self.cls_pw = cls_pw
        self.obj_pw = obj_pw
        self.giou = giou
        self.obj = obj
        self.cls = cls
        self.classes_num = self.model.net.module.num_classes

    def forward(self, model_output, targets):

        if isinstance(model_output, tuple) and len(model_output) == 2:
            # in test/eval mode the Yolo v3 model output a tuple where the second item is the raw predictions
            _, predictions = model_output
        else:
            predictions = model_output

        detection_targets = build_detection_targets(self.model.net.module, targets)

        float_tensor = torch.cuda.FloatTensor if predictions[0].is_cuda else torch.Tensor
        class_loss, giou_loss, objectness_loss = float_tensor([0]), float_tensor([0]), float_tensor([0])

        target_class, target_box, indices, anchor_vec = detection_targets
        reduction = 'mean'  # Loss reduction (sum or mean)

        # DEFINE CRITERIA
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=float_tensor([self.cls_pw]), reduction=reduction)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=float_tensor([self.obj_pw]), reduction=reduction)

        # COMPUTE THE LOSSES BASED ON EACH ONE OF THE YOLO LAYERS PREDICTIONS
        grid_points_num, targets_num = 0, 0
        for yolo_layer_index, yolo_layer_prediction in enumerate(predictions):
            image, anchor, grid_y, grid_x = indices[yolo_layer_index]
            target_object = torch.zeros_like(yolo_layer_prediction[..., 0])
            grid_points_num += target_object.numel()

            # COMPUTE LOSSES
            nb = len(image)
            if nb:  # number of targets
                targets_num += nb
                predictions_for_targets = yolo_layer_prediction[image, anchor, grid_y, grid_x]
                target_object[image, anchor, grid_y, grid_x] = 1.0

                # GIoU LOSS CALCULATION
                pxy = torch.sigmoid(
                    predictions_for_targets[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
                bbox_prediction = torch.cat(
                    (pxy, torch.exp(predictions_for_targets[:, 2:4]).clamp(max=1E3) * anchor_vec[yolo_layer_index]), 1)
                giou = 1.0 - calculate_bbox_iou_elementwise(bbox_prediction.t(), target_box[yolo_layer_index],
                                                            x1y1x2y2=False, GIoU=True)
                giou_loss += giou.sum() if reduction == 'sum' else giou.mean()

                # ONLY RELEVANT TO MULTIPLE CLASSES
                if self.classes_num > 1:
                    class_targets = torch.zeros_like(predictions_for_targets[:, 5:])
                    class_targets[range(nb), target_class[yolo_layer_index]] = 1.0
                    class_loss += BCEcls(predictions_for_targets[:, 5:], class_targets)

            objectness_loss += BCEobj(yolo_layer_prediction[..., 4], target_object)

        if reduction == 'sum':
            giou_loss *= 3 / targets_num
            objectness_loss *= 3 / grid_points_num
            class_loss *= 3 / targets_num / self.classes_num

        loss = giou_loss * self.giou + objectness_loss * self.obj + class_loss * self.cls
        return loss, torch.cat((giou_loss, objectness_loss, class_loss, loss)).detach()
