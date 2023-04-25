import requests
import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and YOLO-NAS are supported.
model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

# We want to use cuda if available to speed up inference.
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

video_path = "pose_elephant_flip.mp4"

# Download the video to the local file system.
with open(video_path, mode="wb") as f:
    response = requests.get("https://deci-pretrained-models.s3.amazonaws.com/sample_images/pose_elephant_flip.mp4")
    f.write(response.content)

predictions = model.predict(video_path)
predictions.show()
predictions.save("pose_elephant_flip_prediction.mp4")
predictions.save("pose_elephant_flip_prediction.gif")  # Can also be saved as a gif.
