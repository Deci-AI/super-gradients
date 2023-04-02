from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.YOLOX_L, pretrained_weights="coco")

# IMAGES = [
#     "https://miro.medium.com/v2/resize:fit:500/0*w1s81z-Q72obhE_z",
#     "https://s.hs-data.com/bilder/spieler/gross/128069.jpg",
#     "https://datasets-server.huggingface.co/assets/Chris1/cityscapes/--/Chris1--cityscapes/train/28/image/image.jpg",
# ]
VIDEO_FILE_NAME = "/home/louis.dupont/PycharmProjects/test_video.mp4"
model.predict_video(source_path=VIDEO_FILE_NAME, output_path=VIDEO_FILE_NAME.replace("test_video", "test_videov2"), iou=0.65, conf=0.5)
# prediction =
# prediction.show()
