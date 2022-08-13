from super_gradients.training.datasets.datasets_utils import DetectionMultiscalePrePredictionCallback

DEFAULT_TRANSFORMS = {"mixup_prob": .0,  # probability to apply per-sample mixup
                      "degrees": 0.,  # rotation degrees, randomly sampled from [-degrees, degrees]
                      "shear": 0.,  # shear degrees, randomly sampled from [-degrees, degrees]
                      "flip_prob": 0.,  # probability to apply horizontal flip
                      "hsv_prob": 0.,  # probability to apply HSV transform
                      "hgain": 1,  # HSV transform hue gain (randomly sampled from [-hgain, hgain])
                      "sgain": 1,  # HSV transform saturation gain (randomly sampled from [-sgain, sgain])
                      "vgain": 1,  # HSV transform value gain (randomly sampled from [-vgain, vgain])
                      "mosaic_scale": [0.5, 1.5],  # random rescale range (keeps size by padding/cropping) after mosaic transform.
                      "mixup_scale": [1., 1.],  # random rescale range for the additional sample in mixup
                      "mosaic_prob": 0.5,  # probability to apply mosaic
                      "translate": 0.,  # image translation fraction
                      "filter_box_candidates": False,  # whether to filter out transformed bboxes by edge size, area ratio, and aspect ratio.
                      "wh_thr": 2,  # edge size threshold when filter_box_candidates = True (pixels)
                      "ar_thr": 20,  # aspect ratio threshold when filter_box_candidates = True
                      "area_thr": 0.1}  # threshold for area ratio between original image and the transformed one, when when filter_box_candidates = True


DEFAULT_TRAINING_PARAMS = {"max_epochs": 20,
                           "lr_mode": "cosine",
                           "cosine_final_lr_ratio": 0.04,
                           "initial_lr": 0.01,
                           "lr_warmup_epochs": 5,
                           "warmup_bias_lr": 0.0,
                           "warmup_momentum": 0.9,
                           "lr_cooldown_epochs": 15,
                           "batch_accumulate": 1,
                           "ema": True,
                           "mixed_precision": True,
                           "loss_logging_items_names": ["iou", "obj", "cls", "l1", "num_fg", "loss"],
                           "metric_to_watch": "mAP@0.50:0.95",
                           "greater_metric_to_watch_is_better": True,
                           "forward_pass_prep_fn": DetectionMultiscalePrePredictionCallback(),
                           "loss": "yolox_loss",
                           "criterion_params": {"strides": [8, 16, 32], "num_classes": 80},
                           "optimizer": "SGD",
                           "optimizer_params": {"momentum": 0.2, "weight_decay": 0.0005, "nesterov": True},
                           "valid_metrics_list": [],
                           "phase_callbacks": []}
