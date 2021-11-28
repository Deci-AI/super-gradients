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

from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface

import super_gradients
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.models import HpmStruct
import argparse

from super_gradients.training.metrics import Accuracy, Top5


parser = argparse.ArgumentParser()
super_gradients.init_trainer()

parser.add_argument("--reload", action="store_true")
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--experiment_name", type=str, default="ddrnet_23")
parser.add_argument("-s", "--slim", action="store_true", help='train the slim version of DDRNet23')

args, _ = parser.parse_known_args()
distributed = super_gradients.is_distributed()
devices = torch.cuda.device_count() if not distributed else 1

train_params_ddr = {"max_epochs": args.max_epochs,
                    "lr_mode": "step",
                    "lr_updates": [30, 60, 90],
                    "lr_decay_factor": 0.1,
                    "initial_lr": 0.1 * devices,
                    "optimizer": "SGD",
                    "optimizer_params": {"weight_decay": 0.0001, "momentum": 0.9, "nesterov": True},
                    "loss": "cross_entropy",
                    "train_metrics_list": [Accuracy(), Top5()],
                    "valid_metrics_list": [Accuracy(), Top5()],
                    "loss_logging_items_names": ["Loss"],
                    "metric_to_watch": "Accuracy",
                    "greater_metric_to_watch_is_better": True
                    }

dataset_params = {"batch_size": args.batch,
                  "color_jitter": 0.4,
                  "random_erase_prob": 0.2,
                  "random_erase_value": 'random',
                  "train_interpolation": 'random',
                  "auto_augment_config_string": 'rand-m9-mstd0.5'
                  }

model = SgModel(experiment_name=args.experiment_name,
                multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL if distributed else MultiGPUMode.DATA_PARALLEL,
                device='cuda')

dataset = ImageNetDatasetInterface(dataset_params=dataset_params)

model.connect_dataset_interface(dataset, data_loader_num_workers=8 * devices)

arch_params = HpmStruct(**{"num_classes": 1000, "aux_head": False, "classification_mode": True, 'dropout_prob': 0.3})

model.build_model(architecture="ddrnet_23_slim" if args.slim else "ddrnet_23",
                  arch_params=arch_params,
                  load_checkpoint=args.reload)
model.train(training_params=train_params_ddr)
