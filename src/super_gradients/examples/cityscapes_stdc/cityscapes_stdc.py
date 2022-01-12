
import torchvision.transforms

import super_gradients
from omegaconf import DictConfig
import hydra
import pkg_resources
from super_gradients.training import StrictLoad
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import CityscapesDatasetInterface
from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CITYSCAPES_IGNORE_LABEL
from torchvision import transforms
from super_gradients.training.utils.segmentation_utils import RandomFlip, RandomRescale, PadShortToCropSize, \
    CropImageAndMask, Rescale, ColorJitterSeg
from super_gradients.training.losses.stdc_loss import STDCLoss
from super_gradients.training.metrics.segmentation_metrics import IoU, PixelAccuracy


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name= "cityscapes_stdc_seg50")
def train(cfg: DictConfig) -> None:
    # INSTANTIATE ALL OBJECTS IN CFG
    cfg = hydra.utils.instantiate(cfg)

    # INIT DATASET_INTERFACE
    color_jitter_args = (cfg.dataset_params.color_jitter,) * 3
    cfg.dataset_params.update({
        "image_mask_transforms_aug": transforms.Compose([
            ColorJitterSeg(*color_jitter_args),
            RandomFlip(),
            RandomRescale(scales=cfg.dataset_params.random_scales),
            PadShortToCropSize(cfg.dataset_params.crop_size, fill_mask=19),
            CropImageAndMask(crop_size=cfg.dataset_params.crop_size, mode="random")
        ]),
        "image_mask_transforms": transforms.Compose([Rescale(scale_factor=cfg.dataset_params.eval_scale)])
    })
    dataset_interface = CityscapesDatasetInterface(dataset_params=cfg.dataset_params)
    num_classes = len(dataset_interface.classes)

    # INITIATE LOSS FUNCTION
    loss = STDCLoss(num_classes=num_classes, ignore_index=CITYSCAPES_IGNORE_LABEL,
                    **cfg.training_hyperparams.criterion_params)
    loss_train_params = loss.get_train_named_params()

    # INITIATE METRICS - IoU metric require num_classes including ignore_index, thus we need to pass `num_classes + 1`
    train_metrics_list = [PixelAccuracy(ignore_label=CITYSCAPES_IGNORE_LABEL),
                          IoU(num_classes=num_classes + 1, ignore_index=CITYSCAPES_IGNORE_LABEL)]
    valid_metrics_list = [PixelAccuracy(ignore_label=CITYSCAPES_IGNORE_LABEL),
                          IoU(num_classes=num_classes + 1, ignore_index=CITYSCAPES_IGNORE_LABEL)]

    cfg.training_hyperparams.update({
        "loss": loss,
        "extra_train_params": loss_train_params,
        "train_metrics_list": train_metrics_list,
        "valid_metrics_list": valid_metrics_list
    })

    cfg.sg_model.connect_dataset_interface(dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)

    if cfg.load_checkpoint:
        cfg.sg_model.build_model(cfg.architecture, arch_params=cfg.arch_params, load_checkpoint=True)
    else:
        cfg.sg_model.build_model(cfg.architecture, arch_params=cfg.arch_params,
                                 strict_load=StrictLoad.NO_KEY_MATCHING,
                                 load_backbone=cfg.load_backbone,
                                 load_weights_only=cfg.load_backbone,
                                 external_checkpoint_path=cfg.external_checkpoint_path)
    # TRAIN
    cfg.sg_model.train(training_params=cfg.training_hyperparams)


if __name__ == "__main__":
    super_gradients.init_trainer()
    train()
