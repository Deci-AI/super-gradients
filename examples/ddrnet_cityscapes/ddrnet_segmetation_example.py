"""
TODO: REFACTOR AS YAML FILES RECIPE

Train DDRNet23 according to the paper

Usage:
    python -m torch.distributed.launch --nproc_per_node=4 ddrnet_segmentation_example.py [-s for slim]
        [-d $n for decinet_$n backbone] --pretrained_bb_path <path>

Training time:
    DDRNet23:           19H (on 4 x 2080Ti)
    DDRNet23 slim:      13H (on 4 x 2080Ti)

Validation mIoU:
    DDRNet23:           78.94 (paper: 79.1±0.3)
    DDRNet23 slim:      76.79 (paper: 77.3±0.4)

Official git repo:
    https://github.com/ydhongHIT/DDRNet

Paper:
    https://arxiv.org/pdf/2101.06085.pdf

Pretained checkpoints:
    Backbones (trained by the original authors):
        s3://deci-model-safe-research/DDRNet/DDRNet23_bb_imagenet.pth
        s3://deci-model-safe-research/DDRNet/DDRNet23s_bb_imagenet.pth
    Segmentation (trained using this recipe:
        s3://deci-model-safe-research/DDRNet/DDRNet23_new/ckpt_best.pth
        s3://deci-model-safe-research/DDRNet/DDRNet23s_new/ckpt_best.pth

Comments:
    * Pretrained backbones were used
    * To pretrain the backbone on imagenet - see ddrnet_classification_example
"""

import torch

from super_gradients.training.metrics.segmentation_metrics import IoU, PixelAccuracy

import super_gradients
from super_gradients.training import SgModel, MultiGPUMode
import argparse
import torchvision.transforms as transforms

from super_gradients.training.utils.segmentation_utils import RandomFlip, PadShortToCropSize, CropImageAndMask, RandomRescale
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss
from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CITYSCAPES_IGNORE_LABEL
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import CityscapesDatasetInterface

parser = argparse.ArgumentParser()

super_gradients.init_trainer()

parser.add_argument("--reload", action="store_true")
parser.add_argument("--max_epochs", type=int, default=485)
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--img_size", type=int, default=1024)
parser.add_argument("--experiment_name", type=str, default="ddrnet_23")
parser.add_argument("--pretrained_bb_path", type=str)
parser.add_argument("-s", "--slim", action="store_true", help='train the slim version of DDRNet23')

args, _ = parser.parse_known_args()
distributed = super_gradients.is_distributed()
devices = torch.cuda.device_count() if not distributed else 1

dataset_params = {
    "batch_size": args.batch,
    "val_batch_size": args.batch,
    "dataset_dir": "/home/ofri/cityscapes/",
    "crop_size": args.img_size,
    "img_size": args.img_size,
    "image_mask_transforms_aug": transforms.Compose([
        # ColorJitterSeg(brightness=0.5, contrast=0.5, saturation=0.5), # TODO - add
        RandomFlip(),
        RandomRescale(scales=(0.5, 2.0)),
        PadShortToCropSize(args.img_size, fill_mask=CITYSCAPES_IGNORE_LABEL,
                           fill_image=(CITYSCAPES_IGNORE_LABEL, 0, 0)),  # Legacy padding color that works best with this recipe
        CropImageAndMask(crop_size=args.img_size, mode="random"),
    ]),
    "image_mask_transforms": transforms.Compose([])  # no transform for evaluation
}

# num_classes for IoU includes the ignore label
train_metrics_list = [PixelAccuracy(ignore_label=CITYSCAPES_IGNORE_LABEL),
                      IoU(num_classes=20, ignore_index=CITYSCAPES_IGNORE_LABEL)]
valid_metrics_list = [PixelAccuracy(ignore_label=CITYSCAPES_IGNORE_LABEL),
                      IoU(num_classes=20, ignore_index=CITYSCAPES_IGNORE_LABEL)]

train_params = {"max_epochs": args.max_epochs,
                "initial_lr": 1e-2,
                "loss": DDRNetLoss(ignore_label=CITYSCAPES_IGNORE_LABEL, num_pixels_exclude_ignored=False),
                "lr_mode": "poly",
                "ema": True,  # unlike the paper (not specified in paper)
                "average_best_models": True,
                "optimizer": "SGD",
                "mixed_precision": False,
                "optimizer_params":
                    {"weight_decay": 5e-4,
                     "momentum": 0.9},
                "train_metrics_list": train_metrics_list,
                "valid_metrics_list": valid_metrics_list,
                "loss_logging_items_names": ["main_loss", "aux_loss", "Loss"],
                "metric_to_watch": "IoU",
                "greater_metric_to_watch_is_better": True
                }

arch_params = {"num_classes": 19, "aux_head": True, "sync_bn": True}

model = SgModel(args.experiment_name,
                multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL if distributed else MultiGPUMode.DATA_PARALLEL,
                device='cuda')

dataset_interface = CityscapesDatasetInterface(dataset_params=dataset_params, cache_labels=False)

model.connect_dataset_interface(dataset_interface, data_loader_num_workers=8 * devices)

model.build_model(architecture="ddrnet_23_slim" if args.slim else "ddrnet_23",
                  arch_params=arch_params,
                  load_checkpoint=args.reload,
                  load_weights_only=args.pretrained_bb_path is not None,
                  load_backbone=args.pretrained_bb_path is not None,
                  external_checkpoint_path=args.pretrained_bb_path)

model.train(training_params=train_params)
