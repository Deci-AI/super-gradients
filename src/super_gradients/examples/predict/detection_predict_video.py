from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")

video_path = "/home/louis.dupont/demo_video_480.mov"
model.predict_video(video_path=video_path)
