from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.pipelines.pipelines import DetectionPipeline


model = models.get(Models.YOLOX_S, pretrained_weights="coco")
model.eval()

# pipe = DetectionPipeline.from_pretrained(model)
prediction = model.predict("https://miro.medium.com/v2/resize:fit:500/0*w1s81z-Q72obhE_z")
prediction.show()

pipe = DetectionPipeline.from_pretrained(model)
prediction2 = pipe("https://s.hs-data.com/bilder/spieler/gross/128069.jpg")
prediction2.show()


print("")
