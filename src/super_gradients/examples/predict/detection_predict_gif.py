from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").cuda()

predictions = model.predict("data/bus.jpg", conf=0.45, iou=0.45)
predictions.show()
