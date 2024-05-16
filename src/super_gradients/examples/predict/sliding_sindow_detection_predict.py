import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models


# Note that currently only YoloX, PPYoloE and YOLO-NAS are supported.
from super_gradients.training.models.detection_models.sliding_window_detection_forward_wrapper import SlidingWindowInferenceDetectionWrapper

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

# We want to use cuda if available to speed up inference.
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

model = SlidingWindowInferenceDetectionWrapper(model=model, tile_size=640, tile_step=160, tile_nms_conf=0.35)

predictions = model.predict(
    "https://images.pexels.com/photos/7968254/pexels-photo-7968254.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", skip_image_resizing=True
)
predictions.show()
predictions.save(output_path="2.jpg")  # Save in working directory
