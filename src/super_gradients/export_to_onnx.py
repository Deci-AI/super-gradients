from super_gradients.common.object_names import Models
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.dataloaders import dataloaders
from super_gradients.training.datasets.data_formats.default_formats import LABEL_CXCYWH
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.datasets.detection_datasets.coco_detection import COCODetectionDataset
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.pre_launch_callbacks import modify_params_for_qat
from super_gradients.training.transforms.transforms import (
    DetectionMosaic,
    DetectionRandomAffine,
    DetectionHSV,
    DetectionHorizontalFlip,
    DetectionPaddedRescale,
    DetectionStandardize,
    DetectionTargetsFormatTransform,
)
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN

trainer = Trainer(experiment_name="yolo_nas_s_coco")
net = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

train_dataset_params = {
    "data_dir": "/home/ofri/PycharmProjects/super-gradients/tests/data/tinycoco/",
    "subdir": "images/train2017",
    "json_file": "instances_train2017.json",
    "input_dim": (640, 640),
    "ignore_empty_annotations": False,
    "transforms": [
        DetectionMosaic(prob=1.0, input_dim=(640, 640)),
        DetectionRandomAffine(degrees=0.0, scales=(0.5, 1.5), shear=0.0, target_size=(640, 640), filter_box_candidates=False, border_value=128),
        DetectionHSV(prob=1.0, hgain=5, vgain=30, sgain=30),
        DetectionHorizontalFlip(prob=0.5),
        DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
        DetectionStandardize(max_value=255),
        DetectionTargetsFormatTransform(max_targets=300, input_dim=(640, 640), output_format=LABEL_CXCYWH),
    ],
}

val_dataset_params = {
    "data_dir": "/home/ofri/PycharmProjects/super-gradients/tests/data/tinycoco/",
    "subdir": "images/val2017",
    "json_file": "instances_val2017.json",
    "input_dim": (640, 640),
    "ignore_empty_annotations": False,
    "transforms": [
        DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
        DetectionStandardize(max_value=255),
        DetectionTargetsFormatTransform(max_targets=300, input_dim=(640, 640), output_format=LABEL_CXCYWH),
    ],
}

train_dataloader_params = {
    "shuffle": True,
    "batch_size": 16,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": CrowdDetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed,
    "min_samples": 512,
}

val_dataloader_params = {
    "shuffle": False,
    "batch_size": 1,
    "num_workers": 2,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": CrowdDetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed,
}

train_params = {
    "warmup_initial_lr": 1e-6,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "AdamW",
    "zero_weight_decay_on_bias_and_bn": True,
    "lr_warmup_epochs": 3,
    "warmup_mode": "linear_epoch_step",
    "optimizer_params": {"weight_decay": 0.0001},
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 10,
    "mixed_precision": True,
    "loss": PPYoloELoss(use_static_assigner=False, num_classes=80, reg_max=16),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=80,
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
        )
    ],
    "metric_to_watch": "mAP@0.50",
}

train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params = modify_params_for_qat(
    train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params
)

trainset = COCODetectionDataset(**train_dataset_params)
valset = COCODetectionDataset(**val_dataset_params)

train_loader = dataloaders.get(dataset=trainset, dataloader_params=train_dataloader_params)

valid_loader = dataloaders.get(dataset=valset, dataloader_params=val_dataloader_params)

trainer.qat(model=net, training_params=train_params, train_loader=train_loader, valid_loader=valid_loader, calib_loader=train_loader)
