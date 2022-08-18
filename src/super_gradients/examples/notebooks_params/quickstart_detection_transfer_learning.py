from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.metrics.detection_metrics import DetectionMetrics
from super_gradients.training.utils.detection_utils import DetectionCollateFN

DEFAULT_DATASET_PARAM = {"cache_dir": "",
                         "train_collate_fn": DetectionCollateFN(),
                         "val_collate_fn": DetectionCollateFN(),
                         "tight_box_rotation": True,
                         "class_inclusion_list": None,
                         "train_max_num_samples": None,
                         "val_max_num_samples": None,
                         "cache_train_images": False,
                         "cache_val_images": False,
                         "with_crowd": False}

DEFAULT_TRAINING_PARAMS = {"max_epochs": 50,
                           "lr_mode": "cosine",
                           "initial_lr": 0.0032,
                           "cosine_final_lr_ratio": 0.12,
                           "lr_warmup_epochs": 2,
                           "warmup_bias_lr": 0.05,  # LR TO START FROM DURING WARMUP (DROPS DOWN DURING WARMUP EPOCHS) FOR BIAS.
                           "loss": "yolox_loss",
                           "loss_logging_items_names": ["iou", "obj", "cls", "l1", "num_fg", "Loss"],
                           "criterion_params": {"strides": [8, 16, 32], "num_classes": 1},
                           "optimizer": "SGD",
                           "warmup_momentum": 0.5,
                           "optimizer_params": {"momentum": 0.9, "weight_decay": 0.0001, "nesterov": True},
                           "ema": True,
                           "train_metrics_list": [],
                           "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(iou=0.65, conf=0.01),
                                                                   normalize_targets=True,
                                                                   num_cls=80)],
                           "metric_to_watch": "mAP@0.50:0.95",
                           "greater_metric_to_watch_is_better": True,
                           "phase_callbacks": []}
