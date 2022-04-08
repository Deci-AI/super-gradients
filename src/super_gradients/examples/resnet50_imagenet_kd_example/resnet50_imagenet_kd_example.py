from super_gradients.training.kd_model.kd_model import KDModel
from super_gradients.training.metrics import Accuracy
from super_gradients.training.losses.kd_losses import KDLogitsLoss
from super_gradients.training.losses import LabelSmoothingCrossEntropyLoss
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface
from super_gradients.training import MultiGPUMode
import super_gradients

super_gradients.init_trainer()

train_params = {"max_epochs": 400,
                "lr_mode": "cosine",
                "lr_warmup_epochs": 5,
                "initial_lr": 0.1,
                "ema": False,
                "save_ckpt_epoch_list": [50, 100, 150, 200, 300],
                "mixed_precision": True,
                "loss": KDLogitsLoss(LabelSmoothingCrossEntropyLoss(), distillation_loss_coeff=0.8),
                "optimizer": "SGD",
                "train_metrics_list": [Accuracy()],
                "valid_metrics_list": [Accuracy()],
                "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True,
                "loss_logging_items_names": ["Loss", "Task Loss", "Distillation Loss"],
                "average_best_models": True, "zero_weight_decay_on_bias_and_bn": True, "batch_accumulate": 2}

dataset_params = {"resize_size": 249,
                  "batch_size": 32,
                  "random_erase_prob": 0,
                  "random_erase_value": "random",
                  "train_interpolation": "random",
                  "rand_augment_config_string": "rand-m7-mstd0.5",
                  "cutmix": True,
                  "cutmix_params": {"mixup_alpha": 0.2,
                                    "cutmix_alpha": 1.0,
                                    "label_smoothing": 0.1
                                    },
                  "img_mean": [0.5, 0.5, 0.5],
                  "img_std": [0.5, 0.5, 0.5]
                  }

kd_model = KDModel("resnet18_imagenet_kd_with_vit_large_teacher",
                   multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL
                   , device='cuda')

dataset = ImageNetDatasetInterface(dataset_params=dataset_params)
kd_model.connect_dataset_interface(dataset, data_loader_num_workers=8)

kd_model.build_model(student_architecture='resnet18',
                     teacher_architecture='vit_large',
                     student_arch_params={'num_classes': 1000, "droppath_prob": 0.05},
                     teacher_arch_params={'num_classes': 1000, "image_size": [224, 224],
                                          "patch_size": [16, 16]},
                     checkpoint_params={'teacher_pretrained_weights': "imagenet"})

kd_model.train(training_params=train_params)
