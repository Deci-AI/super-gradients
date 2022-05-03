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
from super_gradients.training.datasets import CoCoDetectionDatasetInterface, CoCo2014DetectionDatasetInterface
from super_gradients.training.models.yolov5_base import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.datasets.datasets_utils import ComposedCollateFunction, MultiScaleCollateFunction
from super_gradients.common.aws_connection.aws_secrets_manager_connector import AWSSecretsManagerConnector
from super_gradients.training.metrics import DetectionMetrics

super_gradients.init_trainer()

parser = argparse.ArgumentParser()

#################################
# Model Options
################################

parser.add_argument("--model", type=str, required=True, choices=["s", "m", "l", "x", "c"],
                    help='on of s,m,l,x,c (small, medium, large, extra-large, custom)')
parser.add_argument("--depth", type=float, help='not applicable for default models(s/m/l/x)')
parser.add_argument("--width", type=float, help='not applicable for default models(s/m/l/x)')
parser.add_argument("--reload", action="store_true")
parser.add_argument("--max_epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--test-img-size", type=int, default=320)
parser.add_argument("--train-img-size", type=int, default=640)
parser.add_argument("--multi-scale", action="store_true")
parser.add_argument("--coco2014", action="store_true")

args, _ = parser.parse_known_args()

models_dict = {"s": "yolo_v5s", "m": "yolo_v5m", "l": "yolo_v5l", "x": "yolo_v5x", "c": "custom_yolov5"}

if args.model == "c":
    assert args.depth is not None and args.width is not None, "when setting model type to c (custom), depth and width flags must be set"
    assert 0 <= args.depth <= 1, "depth must be in the range [0,1]"
    assert 0 <= args.width <= 1, "width must be in the range [0,1]"
else:
    assert args.depth is None and args.width is None, "depth and width flags have no effect when the model is not c"

args.model = models_dict[args.model]
distributed = super_gradients.is_distributed()

if args.multi_scale:
    train_collate_fn = ComposedCollateFunction([base_detection_collate_fn,
                                                MultiScaleCollateFunction(target_size=args.train_img_size)])
else:
    train_collate_fn = base_detection_collate_fn

dataset_params = {"batch_size": args.batch,
                  "test_batch_size": args.batch,
                  "train_image_size": args.train_img_size,
                  "test_image_size": args.test_img_size,
                  "test_collate_fn": base_detection_collate_fn,
                  "train_collate_fn": train_collate_fn,
                  "test_sample_loading_method": "default",  # TODO: remove when fixing distributed_data_parallel
                  "dataset_hyper_param": {
                      "hsv_h": 0.015,  # IMAGE HSV-Hue AUGMENTATION (fraction)
                      "hsv_s": 0.7,  # IMAGE HSV-Saturation AUGMENTATION (fraction)
                      "hsv_v": 0.4,  # IMAGE HSV-Value AUGMENTATION (fraction)
                      "degrees": 0.0,  # IMAGE ROTATION (+/- deg)
                      "translate": 0.1,  # IMAGE TRANSLATION (+/- fraction)
                      "scale": 0.5,  # IMAGE SCALE (+/- gain)
                      "shear": 0.0}  # IMAGE SHEAR (+/- deg)
                  }

arch_params = {"depth_mult_factor": args.depth,
               "width_mult_factor": args.width
               }
dataset_string = 'coco2017' if not args.coco2014 else 'coco2014'
model_repo_bucket_name = AWSSecretsManagerConnector.get_secret_value_for_secret_key(aws_env='research',
                                                                                    secret_name='training_secrets',
                                                                                    secret_key='S3.MODEL_REPOSITORY_BUCKET_NAME')
model = SgModel(args.model + '____' + dataset_string,
                model_checkpoints_location="s3://" + model_repo_bucket_name,
                multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL if distributed else MultiGPUMode.DATA_PARALLEL,
                post_prediction_callback=YoloV5PostPredictionCallback())

devices = torch.cuda.device_count() if not distributed else 1

dataset_interface_class = CoCoDetectionDatasetInterface if not args.coco2014 else CoCo2014DetectionDatasetInterface
dataset_interface = dataset_interface_class(dataset_params=dataset_params)
model.connect_dataset_interface(dataset_interface, data_loader_num_workers=20)
model.build_model(args.model, arch_params=arch_params, load_checkpoint=args.reload)

post_prediction_callback = YoloV5PostPredictionCallback()
training_params = {"max_epochs": args.max_epochs,
                   "lr_mode": "cosine",
                   "initial_lr": 0.01,
                   "cosine_final_lr_ratio": 0.2,
                   "lr_warmup_epochs": 3,
                   "batch_accumulate": 1,
                   "warmup_bias_lr": 0.1,
                   "loss": "yolo_v5_loss",
                   "criterion_params": {"model": model},
                   "optimizer": "SGD",
                   "warmup_momentum": 0.8,
                   "optimizer_params": {"momentum": 0.937,
                                        "weight_decay": 0.0005 * (args.batch / 64.0),
                                        "nesterov": True},
                   "mixed_precision": False,
                   "ema": True,
                   "train_metrics_list": [],
                   "valid_metrics_list": [DetectionMetrics(post_prediction_callback=post_prediction_callback,
                                                           num_cls=len(
                                                               dataset_interface.coco_classes))],
                   "loss_logging_items_names": ["GIoU", "obj", "cls", "Loss"],
                   "metric_to_watch": "mAP@0.50:0.95",
                   "greater_metric_to_watch_is_better": True}

print(f"Training Yolo v5 {args.model} on {dataset_string.upper()}:\n width-mult={args.width}, depth-mult={args.depth}, "
      f"train-img-size={args.train_img_size}, test-img-size={args.test_img_size} ")
model.train(training_params=training_params)
