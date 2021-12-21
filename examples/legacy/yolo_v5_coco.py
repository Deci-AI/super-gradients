# Yolo v5 Detection training on CoCo2017 Dataset:
# Yolo v5s train on 320x320 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~28.4
# Yolo v5s train in 640x640 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~29.1

# Yolo v5 Detection training on CoCo2014 Dataset:
# Yolo v5s train on 320x320 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~28.77

# batch size may need to change depending on model size and GPU (2080Ti, V100)
# The code is optimized for running with a Mini-Batch of 64 examples... So depending on the amount of GPUs,
# you should change the "batch_accumulate" param in the training_params dict to be batch_size * gpu_num * batch_accumulate = 64.

import super_gradients
import argparse
import torch
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import \
    PascalVOC2012DetectionDataSetInterface
from super_gradients.training.models.detection_models.yolov5 import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.datasets.datasets_utils import ComposedCollateFunction, MultiScaleCollateFunction
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.utils.detection_utils import Anchors

# torch.backends.cudnn.benchmark = True

super_gradients.init_trainer()

parser = argparse.ArgumentParser()

#################################
# Model Options
################################


parser.add_argument("--depth", type=float, help='not applicable for default models(s/m/l/x)')
parser.add_argument("--width", type=float, help='not applicable for default models(s/m/l/x)')
parser.add_argument("--reload", action="store_true")
parser.add_argument("--max_epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--test-img-size", type=int, default=320)
parser.add_argument("--train-img-size", type=int, default=320)
parser.add_argument("--multi-scale", action="store_true")
parser.add_argument("--coco2014", action="store_true")

args, _ = parser.parse_known_args()

distributed = super_gradients.is_distributed()

if args.multi_scale:
    train_collate_fn = ComposedCollateFunction([base_detection_collate_fn,
                                                MultiScaleCollateFunction(target_size=args.train_img_size)])
else:
    train_collate_fn = base_detection_collate_fn

dataset_params = {"batch_size": 64,
                  "val_batch_size": 64,
                  "train_image_size": args.train_img_size,
                  "val_image_size": args.test_img_size,
                  "val_collate_fn": base_detection_collate_fn,
                  "train_collate_fn": train_collate_fn,
                  "train_sample_loading_method": "mosaic",
                  "val_sample_loading_method": "rectangular",
                  "dataset_hyper_param": {
                      "hsv_h": 0.0138,  # IMAGE HSV-Hue AUGMENTATION (fraction)
                      "hsv_s": 0.664,  # IMAGE HSV-Saturation AUGMENTATION (fraction)
                      "hsv_v": 0.464,  # IMAGE HSV-Value AUGMENTATION (fraction)
                      "degrees": 0.373,  # IMAGE ROTATION (+/- deg)
                      "translate": 0.245,  # IMAGE TRANSLATION (+/- fraction)
                      "scale": 0.898,  # IMAGE SCALE (+/- gain)
                      "shear": 0.602}  # IMAGE SHEAR (+/- deg)
                  }

arch_params = {"pretrained_weights": "coco"}
model = SgModel("pascal_train",
                multi_gpu=MultiGPUMode.OFF,
                post_prediction_callback=YoloV5PostPredictionCallback())

dataset_interface = PascalVOC2012DetectionDataSetInterface(dataset_params=dataset_params)
model.connect_dataset_interface(dataset_interface, data_loader_num_workers=20)
model.build_model("yolo_v5s", arch_params=arch_params, load_checkpoint=args.reload)

post_prediction_callback = YoloV5PostPredictionCallback()
training_params = {"max_epochs": args.max_epochs,
                   "lr_mode": "cosine",
                   "initial_lr": 0.0032,
                   "cosine_final_lr_ratio": 0.12,
                   "lr_warmup_epochs": 2,
                   "batch_accumulate": 1,
                   "warmup_bias_lr": 0.1,
                   "loss": "yolo_v5_loss",
                   "criterion_params": {"anchors": Anchors(
                       anchors_list=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                                     [116, 90, 156, 198, 373, 326]], strides=[8, 16, 32]),
                                        "box_loss_gain": 0.0296,
                                        "cls_loss_gain": 0.243,
                                        "cls_pos_weight": 0.631,
                                        "obj_loss_gain": 0.301,
                                        "obj_pos_weight": 0.911}
    ,
                   "optimizer": "SGD",
                   "warmup_momentum": 0.8,
                   "optimizer_params": {"momentum": 0.843,
                                        "weight_decay": 0.00036,
                                        "nesterov": True},
                   "mixed_precision": False,
                   "ema": True,
                   "train_metrics_list": [],
                   "valid_metrics_list": [DetectionMetrics(post_prediction_callback=post_prediction_callback,
                                                           num_cls=len(
                                                               dataset_interface.classes))],
                   "loss_logging_items_names": ["GIoU", "obj", "cls", "Loss"],
                   "metric_to_watch": "mAP@0.50:0.95",
                   "greater_metric_to_watch_is_better": True}

model.train(training_params=training_params)
