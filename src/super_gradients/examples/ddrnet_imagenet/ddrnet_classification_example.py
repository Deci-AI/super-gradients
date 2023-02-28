"""
TODO: REFACTOR AS YAML FILES RECIPE

Train DDRNet23 backbone in ImageNet according to the paper

Training backbone on imagenet:
python -m torch.distributed.launch --nproc_per_node=4 ddrnet_segmentation_example.py [-s for slim]
    [-d $n for decinet_$n backbone] --train_imagenet

Official git repo:  https://github.com/ydhongHIT/DDRNet
Paper:              https://arxiv.org/pdf/2101.06085.pdf

"""

import torch

from super_gradients.common import MultiGPUMode
from super_gradients.common.object_names import Models
from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation
from torchvision.transforms import RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
import super_gradients
from super_gradients.training import Trainer, models, dataloaders
import argparse
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.datasets.data_augmentation import RandomErase

parser = argparse.ArgumentParser()
super_gradients.init_trainer()

parser.add_argument("--reload", action="store_true")
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--experiment_name", type=str, default="ddrnet_23")
parser.add_argument("-s", "--slim", action="store_true", help="train the slim version of DDRNet23")

args, _ = parser.parse_known_args()
distributed = super_gradients.is_distributed()
devices = torch.cuda.device_count() if not distributed else 1

train_params_ddr = {
    "max_epochs": args.max_epochs,
    "lr_mode": "step",
    "lr_updates": [30, 60, 90],
    "lr_decay_factor": 0.1,
    "initial_lr": 0.1 * devices,
    "optimizer": "SGD",
    "optimizer_params": {"weight_decay": 0.0001, "momentum": 0.9, "nesterov": True},
    "loss": "cross_entropy",
    "train_metrics_list": [Accuracy(), Top5()],
    "valid_metrics_list": [Accuracy(), Top5()],
    "metric_to_watch": "Accuracy",
    "greater_metric_to_watch_is_better": True,
}

dataset_params = {
    "batch_size": args.batch,
    "color_jitter": 0.4,
    "random_erase_prob": 0.2,
    "random_erase_value": "random",
    "train_interpolation": "random",
}


train_transforms = [
    RandomResizedCropAndInterpolation(size=224, interpolation="random"),
    RandomHorizontalFlip(),
    ColorJitter(0.4, 0.4, 0.4),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomErase(0.2, "random"),
]

trainer = Trainer(
    experiment_name=args.experiment_name, multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL if distributed else MultiGPUMode.DATA_PARALLEL, device="cuda"
)

train_loader = dataloaders.imagenet_train(dataset_params={"transforms": train_transforms}, dataloader_params={"batch_size": args.batch})
valid_loader = dataloaders.imagenet_val()

model = models.get(
    Models.DDRNET_23_SLIM if args.slim else Models.DDRNET_23,
    arch_params={"use_aux_heads": False, "classification_mode": True, "dropout_prob": 0.3},
    num_classes=1000,
)

trainer.train(model=model, training_params=train_params_ddr, train_loader=train_loader, valid_loader=valid_loader)
