import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models


model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

model.prep_model_for_conversion(input_size=(640, 640))
onnx_input = torch.zeros((1, 3, 640, 640))

torch.onnx.export(model, onnx_input, f="yolo_nas_l.onnx")
