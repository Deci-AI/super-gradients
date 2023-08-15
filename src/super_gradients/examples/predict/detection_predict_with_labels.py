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

# the loaded images are np.ndarrays images of shape (H,W,3)
# the loaded targets are np.ndarrays of shape (num_boxes,x1,y1,x2,y2,class_id)
image1, target1, _ = dataset[0]
image2, target2, _ = dataset[1]

# images from COCODetectionDataset are RGB and images as np.ndarrays are expected to be BGR
image2 = image2[:, :, ::-1]
image1 = image1[:, :, ::-1]

predictions = model.predict([image1, image2])
predictions.show(target_bboxes=[target1[:, :4], target2[:, :4]], target_class_ids=[target1[:, 4], target2[:, 4]], target_bboxes_format="xyxy")
predictions.save(
    output_folder="", target_bboxes=[target1[:, :4], target2[:, :4]], target_class_ids=[target1[:, 4], target2[:, 4]], target_bboxes_format="xyxy"
)  # Save in working directory
