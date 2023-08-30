from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17, checkpoint_path="ckpt_best_pcgj2jlh.pth")
model.predict("https://deci-datasets-research.s3.amazonaws.com/image_samples/beatles-abbeyroad.jpg").show()
