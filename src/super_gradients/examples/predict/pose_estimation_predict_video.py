import requests
import torch
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and YOLO-NAS are supported.
model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")

# We want to use cuda if available to speed up inference.
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

video_path = "slickback.mp4"

# Download the video to the local file system.
with open(video_path, mode="wb") as f:
    response = requests.get("https://deci-pretrained-models.s3.amazonaws.com/sample_images/slickback.mp4")
    f.write(response.content)

predictions = model.predict(video_path)
predictions.save("slickback_prediction.mp4")

predictions = model.predict(video_path)
predictions.save("slickback_prediction.gif")  # Can also be saved as a gif.

predictions = model.predict(video_path)
predictions.show()
