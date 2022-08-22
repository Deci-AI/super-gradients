from super_gradients.training.utils.detection_utils import DetectionCollateFN

DEFAULT_DATASET_PARAM = {
    "train_collate_fn": DetectionCollateFN(),
    "val_collate_fn": DetectionCollateFN(),
    "tight_box_rotation": True,
    "class_inclusion_list": None,
    "train_max_num_samples": None,
    "val_max_num_samples": None,
    "cache_train_images": False,
    "cache_val_images": False,
    "with_crowd": False
}

DEFAULT_TRAINING_PARAMS = {"max_epochs": 20,
                           "lr_mode": "cosine",
                           "cosine_final_lr_ratio": 0.04,
                           "initial_lr": 0.01,
                           "lr_warmup_epochs": 5,
                           "warmup_bias_lr": 0.0,
                           "warmup_momentum": 0.9,
                           "lr_cooldown_epochs": 2,
                           "batch_accumulate": 1,
                           "ema": True,
                           "mixed_precision": True,
                           "loss_logging_items_names": ["iou", "obj", "cls", "l1", "num_fg", "loss"],
                           "metric_to_watch": "mAP@0.50:0.95",
                           "greater_metric_to_watch_is_better": True,
                           "loss": "yolox_loss",
                           "criterion_params": {"strides": [8, 16, 32], "num_classes": 80},
                           "optimizer": "SGD",
                           "optimizer_params": {"momentum": 0.2, "weight_decay": 0.0005, "nesterov": True},
                           "valid_metrics_list": [],
                           "phase_callbacks": []}
