from super_gradients.common.object_names import Models
from super_gradients.training import models


model = models.get(Models.YOLOX_S, pretrained_weights="coco")
model.eval()

SEG_IMAGE = "https://datasets-server.huggingface.co/assets/Chris1/cityscapes/--/Chris1--cityscapes/train/28/image/image.jpg"

DET_IMAGE1 = "https://miro.medium.com/v2/resize:fit:500/0*w1s81z-Q72obhE_z"
DET_IMAGE2 = "https://s.hs-data.com/bilder/spieler/gross/128069.jpg"


prediction = model.predict(SEG_IMAGE, iou=0.655, conf=0.01)
prediction.show()


prediction = model.predict(DET_IMAGE1, iou=0.655, conf=0.01)
prediction.show()

prediction = model.predict(DET_IMAGE2, iou=0.655, conf=0.01)
prediction.show()


print("")
