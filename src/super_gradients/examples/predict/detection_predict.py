from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and DeciYolo are supported.
model = models.get(Models.DECIYOLO_L, pretrained_weights="coco")

IMAGES = [
    "../../../../documentation/source/images/examples/countryside.jpg",
    "../../../../documentation/source/images/examples/street_busy.jpg",
    "https://cdn-attachments.timesofmalta.com/cc1eceadde40d2940bc5dd20692901371622153217-1301777007-4d978a6f-620x348.jpg",
]

prediction = model.predict(IMAGES)
prediction.show()
