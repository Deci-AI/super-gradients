from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and YOLO-NAS are supported.
from super_gradients.training.datasets import COCODetectionDataset

model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

IMAGES = [
    "../../../../documentation/source/images/examples/countryside.jpg",
    "../../../../documentation/source/images/examples/street_busy.jpg",
    "https://cdn-attachments.timesofmalta.com/cc1eceadde40d2940bc5dd20692901371622153217-1301777007-4d978a6f-620x348.jpg",
]
dataset = COCODetectionDataset(
    data_dir="/data/coco", subdir="images/val2017", json_file="instances_val2017.json", input_dim=None, transforms=[], cache_annotations=False
)
x, y, _ = dataset[0]
x = x[:, :, ::-1]

predictions = model.predict(x, target_bboxes=y[:, :4], target_class_ids=y[:, 4], target_bboxes_format="xyxy")
predictions.show()
predictions.save(output_folder="")  # Save in working directory
