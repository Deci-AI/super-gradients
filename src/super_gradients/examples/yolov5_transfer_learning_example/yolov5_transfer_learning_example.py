"""
Transfer learning example- from our pretrained YoloV5m (on coco) on Pascal VOC. Reaches 68.8 mAP:0.50:0.95.
"""

import super_gradients
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import \
    PascalVOCUnifiedDetectionDataSetInterface
from super_gradients.training.models.detection_models.yolov5 import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.utils.detection_utils import Anchors

super_gradients.init_trainer()

distributed = super_gradients.is_distributed()

# DEFINE DATASET PARAMS FOR PASCAL VOC
dataset_params = {"batch_size": 48,
                  "val_batch_size": 48,
                  "train_image_size": 512,
                  "val_image_size": 512,
                  "val_collate_fn": base_detection_collate_fn,
                  "train_collate_fn": base_detection_collate_fn,
                  "train_sample_loading_method": "mosaic",
                  "val_sample_loading_method": "default",
                  "dataset_hyper_param": {
                      "hsv_h": 0.0138,  # IMAGE HSV-Hue AUGMENTATION (fraction)
                      "hsv_s": 0.664,  # IMAGE HSV-Saturation AUGMENTATION (fraction)
                      "hsv_v": 0.464,  # IMAGE HSV-Value AUGMENTATION (fraction)
                      "degrees": 0.373,  # IMAGE ROTATION (+/- deg)
                      "translate": 0.245,  # IMAGE TRANSLATION (+/- fraction)
                      "scale": 0.898,  # IMAGE SCALE (+/- gain)
                      "shear": 0.602,
                      "mixup": 0.243  # MIXUP PROBABILITY
                  }
                  }

# INITIALIZE SG MODEL INSTANCE, AND A PASCAL VOC DATASET INTERFACE
model = SgModel("yolov5m_pascal_finetune_augment_fix")
dataset_interface = PascalVOCUnifiedDetectionDataSetInterface(dataset_params=dataset_params, cache_labels=True,
                                                              cache_images=True)

# CONNECTING THE DATASET INTERFACE WILL SET SGMODEL'S CLASSES ATTRIBUTE ACCORDING TO PASCAL VOC
model.connect_dataset_interface(dataset_interface, data_loader_num_workers=8)

# THIS IS WHERE THE MAGIC HAPPENS- SINCE SGMODEL'S CLASSES ATTRIBUTE WAS SET TO BE DIFFERENT FROM COCO'S, AFTER
# LOADING THE PRETRAINED YOLO_V5M, IT WILL CALL IT'S REPLACE_HEAD METHOD AND CHANGE IT'S DETECT LAYER ACCORDING
# TO PASCAL VOC CLASSES
model.build_model("yolo_v5m", arch_params={"pretrained_weights": "coco"})

# WE NOW TUNE THE 3 NORMALIZERS ACCORDING TO THE NEW DATASET ATTRIBUTES,
network = model.net
network = network.module if hasattr(network, 'module') else network
num_levels = network._head._modules_list[-1].detection_layers_num
train_image_size = dataset_params["train_image_size"]

num_branches_norm = 3. / num_levels
num_classes_norm = len(model.classes) / 80.
image_size_norm = train_image_size / 640.

# DEFINE TRAINING PARAMS. SEE DOCS FOR THE FULL LIST.
training_params = {"max_epochs": 50,
                   "lr_mode": "cosine",
                   "initial_lr": 0.0032,
                   "cosine_final_lr_ratio": 0.12,
                   "lr_warmup_epochs": 2,
                   "warmup_bias_lr": 0.05,  # LR TO START FROM DURING WARMUP (DROPS DOWN DURING WARMUP EPOCHS) FOR BIAS.
                   "loss": "yolo_v5_loss",
                   "criterion_params": {"anchors": Anchors(
                       anchors_list=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                                     [116, 90, 156, 198, 373, 326]], strides=[8, 16, 32]),  # MODEL'S ANCHORS
                       "box_loss_gain": 0.0296 * num_branches_norm,  # COEF FOR BOX LOSS COMPONENT, NORMALIZED
                       "cls_loss_gain": 0.243 * num_classes_norm * num_branches_norm,  # COEF FOR CLASSIFICATION
                                                                                       # LOSS COMPONENT, NORMALIZED
                       "cls_pos_weight": 0.631,  # CLASSIFICATION BCE POSITIVE CLASS WEIGHT
                       "obj_loss_gain": 0.301 * image_size_norm ** 2 * num_branches_norm,  # OBJECT BCE COEF, NORMALIZED
                       "obj_pos_weight": 0.911,  # OBJECT BCE POSITIVE CLASS WEIGHT
                       "anchor_threshold": 2.91  # RATIO DEFINING THE SIZE RANGE OF AN ANCHOR.
                   },
                   "optimizer": "SGD",
                   "warmup_momentum": 0.5,
                   "optimizer_params": {"momentum": 0.843,
                                        "weight_decay": 0.00036,
                                        "nesterov": True},
                   "ema": True,
                   "train_metrics_list": [],
                   "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloV5PostPredictionCallback(),
                                                           num_cls=len(
                                                               dataset_interface.classes))],
                   "loss_logging_items_names": ["GIoU", "obj", "cls", "Loss"],
                   "metric_to_watch": "mAP@0.50:0.95",
                   "greater_metric_to_watch_is_better": True,
                   "warmup_mode": "yolov5_warmup"}

# FINALLY, CALL TRAIN
model.train(training_params=training_params)
