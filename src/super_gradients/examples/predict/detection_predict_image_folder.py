from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")

image_folder_path = "/home/louis.dupont/data"
output_folder_path = "/home/louis.dupont/data_out"
model.predict_image_folder(image_folder_path=image_folder_path, output_folder_path=output_folder_path, conf=0.7, batch_size=32)
