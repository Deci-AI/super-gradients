from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.DECIYOLO_L, pretrained_weights="coco")

IMAGES = [
    "https://miro.medium.com/v2/resize:fit:500/0*w1s81z-Q72obhE_z",
    "https://datasets-server.huggingface.co/assets/Chris1/cityscapes/--/Chris1--cityscapes/train/28/image/image.jpg",
    "https://media.timeout.com/images/105921973/750/422/image.jpg",
]
prediction = model.predict(IMAGES)
prediction.show()
