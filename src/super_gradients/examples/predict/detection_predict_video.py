from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and DeciYolo are supported.
model = models.get(Models.DECIYOLO_L, pretrained_weights="coco")

video_path = "<path/to/your/video>"
predictions = model.predict(video_path)
predictions.show()
