from super_gradients.training.metrics.detection_metrics import DetectionMetrics

from super_gradients.training.metrics import DetectionMetricsV2
from super_gradients.training.models.detection_models.yolov5_base import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionCollateFN, CrowdDetectionCollateFN
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import CocoDetectionDatasetInterfaceV2
from super_gradients.training.sg_model import SgModel
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat

dataset_params = {"data_dir": "/data/coco",  # root path to coco data
                  "train_subdir": "images/train2017",  # sub directory path of data_dir containing the train data.
                  "val_subdir": "images/val2017",  # sub directory path of data_dir containing the validation data.
                  "train_json_file": "instances_train2017.json",
                  # path to coco train json file, data_dir/annotations/train_json_file.
                  "val_json_file": "instances_val2017.json",
                  # path to coco validation json file, data_dir/annotations/val_json_file.

                  "batch_size": 16,  # batch size for trainset in CoCoDetectionDatasetInterface
                  "val_batch_size": 128,  # batch size for valset in CoCoDetectionDatasetInterface
                  "val_image_size": 640,  # image size for valset in CoCoDetectionDatasetInterface
                  "train_image_size": 640,  # image size for trainset in CoCoDetectionDatasetInterface

                  "hgain": 5,
                  "sgain": 30,
                  "vgain": 30,

                  "mixup_prob": 1.0,
                  "degrees": 10.,
                  "shear": 2.0,
                  "flip_prob": 0.5,
                  "hsv_prob": 1.0,
                  "mosaic_scale": [0.1, 2],
                  "mixup_scale": [0.5, 1.5],
                  "mosaic_prob": 1.,
                  "translate": 0.1,

                  "val_collate_fn": CrowdDetectionCollateFN(),

                  "train_collate_fn": DetectionCollateFN(),

                  "cache_dir_path": None,
                  "cache_train_images": False,
                  "cache_val_images": False,
                  "targets_format": DetectionTargetsFormat.LABEL_CXCYWH
                  }

sg_model = SgModel("yoloxs_conversion")
di = CocoDetectionDatasetInterfaceV2(dataset_params=dataset_params)
sg_model.connect_dataset_interface(di)
sg_model.build_model(architecture="yolox_l", checkpoint_params={"pretrained_weights": "coco"})
results = sg_model.test(test_loader=sg_model.valid_loader, test_metrics_list=[DetectionMetrics(normalize_targets=True,
                                                                                                 post_prediction_callback=YoloV5PostPredictionCallback(iou=0.65, conf=0.01),
                                                                                                 num_cls=80)])
print("Ols mAP: "+str(results[2]))


results = sg_model.test(test_loader=sg_model.valid_loader, test_metrics_list=[DetectionMetricsV2(normalize_targets=True,
                                                                                                 post_prediction_callback=YoloV5PostPredictionCallback(iou=0.65, conf=0.01),
                                                                                                 num_cls=80)])
print("New mAP: "+str(results[2]))

