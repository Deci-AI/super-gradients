from super_gradients.training.datasets.dataset_interfaces.dataset_interface import SuperviselyPersonsDatasetInterface
from super_gradients.training.sg_model import SgModel
from super_gradients.training.metrics import BinaryIOU
from super_gradients.training.transforms.transforms import ResizeSeg, RandomFlip, RandomRescale, CropImageAndMask, \
    PadShortToCropSize, ColorJitterSeg
from super_gradients.training.utils.callbacks import BinarySegmentationVisualizationCallback, Phase
from torchvision import transforms

# DEFINE DATA TRANSFORMATIONS
dataset_params = {
    "image_mask_transforms_aug": transforms.Compose([ColorJitterSeg(brightness=0.5, contrast=0.5, saturation=0.5),
                                                     RandomFlip(),
                                                     RandomRescale(scales=[0.25, 1.]),
                                                     PadShortToCropSize([320, 480]),
                                                     CropImageAndMask(crop_size=[320, 480],
                                                                      mode="random")]),
    "image_mask_transforms": transforms.Compose([ResizeSeg(h=480, w=320)])
}

dataset_interface = SuperviselyPersonsDatasetInterface(dataset_params)

model = SgModel("regseg48_transfer_learning_old_dice_diff_lrs_head_fixed_50_epochs")

# CONNECTING THE DATASET INTERFACE WILL SET SGMODEL'S CLASSES ATTRIBUTE ACCORDING TO SUPERVISELY
model.connect_dataset_interface(dataset_interface)

# THIS IS WHERE THE MAGIC HAPPENS- SINCE SGMODEL'S CLASSES ATTRIBUTE WAS SET TO BE DIFFERENT FROM CITYSCAPES'S, AFTER
# LOADING THE PRETRAINED REGSET, IT WILL CALL IT'S REPLACE_HEAD METHOD AND CHANGE IT'S SEGMENTATION HEAD LAYER ACCORDING
# TO OUR BINARY SEGMENTATION DATASET
model.build_model("regseg48", arch_params={"pretrained_weights": "cityscapes"})

# DEFINE TRAINING PARAMS. SEE DOCS FOR THE FULL LIST.
train_params = {"max_epochs": 50,
                "lr_mode": "cosine",
                "initial_lr": 0.0064,  # for batch_size=16
                "optimizer_params": {"momentum": 0.843,
                                     "weight_decay": 0.00036,
                                     "nesterov": True},

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
                "loss_logging_items_names": ["loss"],
                "phase_callbacks": [BinarySegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
                                                                            freq=1,
                                                                            last_img_idx_in_batch=4)],
                }

model.train(train_params)
