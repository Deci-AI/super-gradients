from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.YOLOX_N, pretrained_weights="coco")

image_folder_path = "<path/to/image_folder>"
predictions = model.predict(image_folder_path)
predictions.show()
