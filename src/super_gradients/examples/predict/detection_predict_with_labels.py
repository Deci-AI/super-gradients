from super_gradients.common.object_names import Models
from super_gradients.training import models
from pathlib import Path
from super_gradients.training.datasets import COCODetectionDataset

# Note that currently only YoloX, PPYoloE and YOLO-NAS are supported.
model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")
mini_coco_data_dir = str(Path(__file__).parent.parent.parent.parent.parent / "tests" / "data" / "tinycoco")

dataset = COCODetectionDataset(
    data_dir=mini_coco_data_dir, subdir="images/val2017", json_file="instances_val2017.json", input_dim=None, transforms=[], cache_annotations=False
)

# x's are np.ndarrays images of shape (H,W,3)
# y's are np.ndarrays of shape (num_boxes,x1,y1,x2,y2,class_id)
x1, y1, _ = dataset[0]
x2, y2, _ = dataset[1]

# images from COCODetectionDataset are RGB and images as np.ndarrays are expected to be BGR
x2 = x2[:, :, ::-1]
x1 = x1[:, :, ::-1]

predictions = model.predict([x1, x2], target_bboxes=[y1[:, :4], y2[:, :4]], target_class_ids=[y1[:, 4], y2[:, 4]], target_bboxes_format="xyxy")
predictions.show()
predictions.save(output_folder="")  # Save in working directory
