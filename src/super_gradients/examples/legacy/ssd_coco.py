# SSD Detection training on CoCo Dataset:

import argparse
import torch

from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.datasets import CoCoDetectionDatasetInterface
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.utils.ssd_utils import DefaultBoxes, SSDPostPredictCallback

parser = argparse.ArgumentParser()

#################################
# Model Options
################################

parser.add_argument("--reload", action="store_true")
parser.add_argument("--max_epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=60)
parser.add_argument("--test-img-size", type=int, default=256)
parser.add_argument("--train-img-size", type=int, default=256)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--ema-decay", type=float, default=0.9999)
parser.add_argument("--ema-beta", type=float, default=15)

parser.add_argument("--local_rank", type=int, default=-1)

args, _ = parser.parse_known_args()

distributed = args.local_rank >= 0

dataset_params = {"batch_size": args.batch,
                  "test_batch_size": args.batch,
                  "dataset_dir": "/data/coco/",
                  "train_image_size": args.train_img_size,
                  "test_image_size": args.test_img_size,
                  "test_collate_fn": base_detection_collate_fn,
                  "train_collate_fn": base_detection_collate_fn,
                  "test_sample_loading_method": "default",
                  "labels_offset": 1,  # all labels are offset by 1 (0 is none)
                  "dataset_hyper_param": {
                      "hsv_h": 0.015,  # IMAGE HSV-Hue AUGMENTATION (fraction)
                      "hsv_s": 0.7,  # IMAGE HSV-Saturation AUGMENTATION (fraction)
                      "hsv_v": 0.4,  # IMAGE HSV-Value AUGMENTATION (fraction)
                      "degrees": 0.0,  # IMAGE ROTATION (+/- deg)
                      "translate": 0.1,  # IMAGE TRANSLATION (+/- fraction)
                      "scale": 0.5,  # IMAGE SCALE (+/- gain)
                      "shear": 0.0}  # IMAGE SHEAR (+/- deg)
                  }
dboxes = DefaultBoxes.dboxes256_coco()
arch_params = {"num_classes": 81}  # 80 COCO classes + 1 for None
epoch_metrics_headers = {"Epoch": 0, "gpu_mem": 0.0, "sl1": 0.0, "closs": 0.0, "total": 0.0,
                         "targets": 0, "img_size": 0}
results_titles = ['sl1', 'c-loss', 'Train loss',
                  'Precision', 'Recall', 'mAP@0.5:0.95', 'F1', 'val sl1', 'val c-loss',
                  'val loss']
model = SgModel(f'ssd_mobilenet_alpha{args.alpha:.1f}_decay{args.ema_decay:.4E}_beta{args.ema_beta:.2E}',
                model_checkpoints_location="local",
                post_prediction_callback=SSDPostPredictCallback(dboxes=dboxes),
                epoch_metric_headers=epoch_metrics_headers,
                results_titles=results_titles
                )

devices = torch.cuda.device_count() if not distributed else 1

coco_dataset_interface = CoCoDetectionDatasetInterface(dataset_params=dataset_params)
model.connect_dataset_interface(coco_dataset_interface, data_loader_num_workers=32)

model.build_model("ssd_mobilenet_v1", arch_params=arch_params, load_checkpoint=args.reload)
training_params = {"max_epochs": args.max_epochs,
                   "lr_mode": "cosine",
                   "initial_lr": 0.01,
                   "batch_accumulate": 1,
                   "cosine_final_lr_ratio": 0.1,
                   "warmup_bias_lr": 0.1,
                   "loss": "ssd_loss",
                   "criterion_params": {"dboxes": dboxes, "alpha": args.alpha},
                   "optimizer": "SGD",
                   "warmup_momentum": 0.8,
                   "optimizer_params": {"momentum": 0.9,
                                        "weight_decay": 0.0005,
                                        "nesterov": True},
                   "mixed_precision": False,
                   "ema": True,
                   "ema_params": {"decay": args.ema_decay,
                                  "beta": args.ema_beta}
                   }

model.train(training_params=training_params)
