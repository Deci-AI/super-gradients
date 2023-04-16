from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and DeciYolo are supported.
model = models.get(Models.DECIYOLO_S, pretrained_weights="coco")
model.predict_webcam()
