"""Note that this is a feature that is only available for a few models."""
from super_gradients.common.object_names import Models
from super_gradients.training import models


model = models.get(Models.YOLOX_L, pretrained_weights="coco")

DET_IMAGE1 = "https://miro.medium.com/v2/resize:fit:500/0*w1s81z-Q72obhE_z"
DET_IMAGE2 = "https://s.hs-data.com/bilder/spieler/gross/128069.jpg"
SEG_IMAGE = "https://datasets-server.huggingface.co/assets/Chris1/cityscapes/--/Chris1--cityscapes/train/28/image/image.jpg"

prediction = model.predict([DET_IMAGE2, DET_IMAGE1, SEG_IMAGE], iou=0.65, conf=0.5)
prediction.show()
