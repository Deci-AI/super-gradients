from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")

IMAGES = [
    "https://miro.medium.com/v2/resize:fit:500/0*w1s81z-Q72obhE_z",
    "https://s.hs-data.com/bilder/spieler/gross/128069.jpg",
    "https://datasets-server.huggingface.co/assets/Chris1/cityscapes/--/Chris1--cityscapes/train/28/image/image.jpg",
]
prediction = model.predict(images=IMAGES, iou=0.65, conf=0.5)
prediction.show()
