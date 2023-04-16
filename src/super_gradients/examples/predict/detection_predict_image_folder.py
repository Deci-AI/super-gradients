from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and DeciYolo are supported.
model = models.get(Models.DECIYOLO_L, pretrained_weights="coco")

image_folder_path = "../../../../documentation/source/images/examples"
predictions = model.predict(image_folder_path)
predictions.show()
