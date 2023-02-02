from super_gradients.common.object_names import Models
from super_gradients.training import models, dataloaders

from super_gradients.training.sg_trainer import Trainer
from super_gradients.training.metrics import BinaryIOU
from super_gradients.training.transforms.transforms import (
    SegResize,
    SegRandomFlip,
    SegRandomRescale,
    SegCropImageAndMask,
    SegPadShortToCropSize,
    SegColorJitter,
)
from super_gradients.training.utils.callbacks import BinarySegmentationVisualizationCallback, Phase

# DEFINE DATA TRANSFORMATIONS

dl_train = dataloaders.supervisely_persons_train(
    dataset_params={
        "transforms": [
            SegColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            SegRandomFlip(),
            SegRandomRescale(scales=[0.25, 1.0]),
            SegPadShortToCropSize([320, 480]),
            SegCropImageAndMask(crop_size=[320, 480], mode="random"),
        ]
    }
)

dl_val = dataloaders.supervisely_persons_val(dataset_params={"transforms": [SegResize(h=480, w=320)]})

trainer = Trainer("regseg48_transfer_learning_old_dice_diff_lrs_head_fixed_50_epochs")

# THIS IS WHERE THE MAGIC HAPPENS- SINCE TRAINER'S CLASSES ATTRIBUTE WAS SET TO BE DIFFERENT FROM CITYSCAPES'S, AFTER
# LOADING THE PRETRAINED REGSET, IT WILL CALL IT'S REPLACE_HEAD METHOD AND CHANGE IT'S SEGMENTATION HEAD LAYER ACCORDING
# TO OUR BINARY SEGMENTATION DATASET
model = models.get(Models.REGSEG48, pretrained_weights="cityscapes", num_classes=1)

# DEFINE TRAINING PARAMS. SEE DOCS FOR THE FULL LIST.
train_params = {
    "max_epochs": 50,
    "lr_mode": "cosine",
    "initial_lr": 0.0064,  # for batch_size=16
    "optimizer_params": {"momentum": 0.843, "weight_decay": 0.00036, "nesterov": True},
    "cosine_final_lr_ratio": 0.1,
    "multiply_head_lr": 10,
    "optimizer": "SGD",
    "loss": "bce_dice_loss",
    "ema": True,
    "zero_weight_decay_on_bias_and_bn": True,
    "average_best_models": True,
    "mixed_precision": False,
    "metric_to_watch": "mean_IOU",
    "greater_metric_to_watch_is_better": True,
    "train_metrics_list": [BinaryIOU()],
    "valid_metrics_list": [BinaryIOU()],
    "phase_callbacks": [BinarySegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END, freq=1, last_img_idx_in_batch=4)],
}

trainer.train(model=model, training_params=train_params, train_loader=dl_train, valid_loader=dl_val)
