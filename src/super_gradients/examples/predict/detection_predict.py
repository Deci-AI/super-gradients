import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and YOLO-NAS are supported.
model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

# We want to use cuda if available to speed up inference.
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

IMAGES = [
    "https://images.pexels.com/photos/7968254/pexels-photo-7968254.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
]

predictions = model.predict(IMAGES, skip_image_resizing=True)
predictions.show()
predictions.save(output_folder="2")  # Save in working directory
