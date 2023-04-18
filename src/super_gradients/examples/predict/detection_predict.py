from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")

IMAGES = [
    "../../../../documentation/source/images/examples/countryside.jpg",
    "../../../../documentation/source/images/examples/street_busy.jpg",
    "https://cdn-attachments.timesofmalta.com/cc1eceadde40d2940bc5dd20692901371622153217-1301777007-4d978a6f-620x348.jpg",
]

predictions = model.predict(IMAGES)
predictions.show()
predictions.save("<output-folder-path>")
