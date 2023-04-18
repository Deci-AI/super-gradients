import requests
import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.YOLOX_N, pretrained_weights="coco")

# We want to use cuda if available to speed up inference.
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

video_path = "pose_elephant_flip.mp4"

# Download the video to the local file system.
with open(video_path, mode="wb") as f:
    response = requests.get("https://deci-pretrained-models.s3.amazonaws.com/sample_images/pose_elephant_flip.mp4")
    f.write(response.content)

predictions = model.predict(video_path)
predictions.show()
predictions.save("output_path.mp4")
