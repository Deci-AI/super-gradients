import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX and PPYoloE are supported.
model = models.get(Models.YOLOX_N, pretrained_weights="coco")

# We want to use cuda if available to speed up inference.
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

predictions = model.predict(
    "../../../../documentation/source/images/examples/pose_elephant_flip.gif",
)
predictions.show()
predictions.save("pose_elephant_flip_prediction.gif")
predictions.save("pose_elephant_flip_prediction.mp4")  # Can also be saved as a mp4 video.
