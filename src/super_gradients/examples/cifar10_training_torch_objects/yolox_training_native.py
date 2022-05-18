"""
Cifar10 training with SuperGradients training with the following initialized torch objects:

    DataLoaders
    Optimizers
    Networks (nn.Module)
    Schedulers
    Loss functions

Main purpose is to demonstrate training in SG with minimal abstraction and maximal flexibility
"""

from omegaconf import DictConfig
import hydra
import pkg_resources
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from super_gradients import SgModel
import super_gradients
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training import MultiGPUMode
import numpy as np
from super_gradients.training.utils.callbacks import YoloXTrainingStageSwitchCallback
from super_gradients.training.utils.callbacks import DetectionVisualizationCallback, Phase
from super_gradients.training.models.detection_models.yolov5_base import YoloV5PostPredictionCallback
from torchvision.transforms import Resize
from super_gradients.training.utils.distributed_training_utils import get_local_rank, get_world_size
from super_gradients.training.transforms.yolox_transforms import TrainTransform, ValTransform

from super_gradients.training.datasets.detection_datasets.coco_detection_yolox import COCODataset, MosaicDetection
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.datasets.samplers.infinite_sampler import InfiniteSampler
from torch.utils.data import BatchSampler, DataLoader
from super_gradients.training.utils.distributed_training_utils import wait_for_the_master
from super_gradients.training.utils.callbacks import YoloXMultiscaleImagesCallback
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.datasets_utils import ComposedCollateFunction, MultiScaleCollateFunction, MultiscaleForwardPassPrepFunction

logger = get_logger(__name__)


class YoloXCollateFN:
    def __init__(self, resolution, target_resolution=None, val=True, max_targets=120):
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.target_resolution = target_resolution or self.resolution
        self.val = val
        self.max_targets = max_targets

    def _pad_targets(self, data):
        for sample_id, sample in enumerate(data):
            if sample[1].shape[0] < self.max_targets:
                boxes = np.zeros((self.max_targets, 5))
                boxes[:sample[1].shape[0], :] = sample[1]
                boxes = np.roll(boxes, 1, axis=1)
                sample = list(sample)
                sample[1] = boxes
                sample = tuple(sample)
                data[sample_id] = sample

    def _final_rescale(self, inputs, targets):
        # logger.info("final res in collate: "+str(self.target_resolution))
        scale_y = self.target_resolution[0] / self.resolution[0]
        scale_x = self.target_resolution[1] / self.resolution[1]
        if scale_x != 1 or scale_y != 1:
            inputs = torch.nn.functional.interpolate(
                inputs, size=(self.target_resolution, self.target_resolution), mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def __call__(self, data):
        if self.val:
            self._pad_targets(data)
        batch = default_collate(data)
        ims = batch[0]
        targets = batch[1]
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # TODO: divide coords properly to support non rectangular inputs
        targets[:, :, 1:] /= self.resolution[0]
        targets_merged = []
        for i in range(targets.shape[0]):
            targets_im = targets[i, :nlabel[i]]
            batch_column = targets.new_ones((targets_im.shape[0], 1)) * i
            targets_merged.append(torch.cat((batch_column, targets_im), 1))
        targets = torch.cat(targets_merged, 0)
        # ims, targets = self._final_rescale(ims, targets)
        return ims, targets


def get_data_loader(cfg, no_aug=False, cache_img=False):
    local_rank = get_local_rank()
    input_size = (cfg.dataset_params.train_image_size, cfg.dataset_params.train_image_size)
    with wait_for_the_master(local_rank):
        dataset = COCODataset(
            data_dir="/data/coco",
            json_file="instances_train2017.json",
            img_size=input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=0.5,
                hsv_prob=1.0),
            cache=cache_img,
            # add_pseudo_labels=False,
            # json_file_pseudo='instances_pseudolabels2017_yoloxx.json',
            # score_threshold=0,
            # tight_box_rotation=False
        )

    dataset = MosaicDetection(
        dataset,
        mosaic=not no_aug,
        img_size=input_size,
        preproc=TrainTransform(
            max_labels=120,
            flip_prob=0.5,
            hsv_prob=1.0),
        degrees=cfg.dataset_params.dataset_hyper_param.degrees,
        translate=cfg.dataset_params.dataset_hyper_param.translate,
        mosaic_scale=(0.1, 2),
        mixup_scale=(0.5, 1.5),
        shear=cfg.dataset_params.dataset_hyper_param.shear,
        enable_mixup=True,
        mosaic_prob=1.,
        mixup_prob=cfg.dataset_params.dataset_hyper_param.mixup,
    )

    sampler = InfiniteSampler(len(dataset), seed=0)

    batch_sampler = BatchSampler(
        sampler=sampler,
        batch_size=cfg.dataset_params.batch_size,
        drop_last=False,
    )

    dataloader_kwargs = {"num_workers": cfg.data_loader_num_workers, "pin_memory": True}
    # dataloader_kwargs = {"num_workers": 0, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    train_loader = DataLoader(dataset, **dataloader_kwargs,
                              collate_fn=ComposedCollateFunction([YoloXCollateFN(cfg.dataset_params.train_image_size, val=False)]))

    return train_loader


def get_eval_loader(cfg, legacy=False):

    valdataset = COCODataset(
        data_dir='/data/coco',
        json_file="instances_val2017.json",
        name="images/val2017",
        img_size=(cfg.dataset_params.val_image_size, cfg.dataset_params.val_image_size),
        preproc=ValTransform(legacy=legacy),
    )

    if cfg.sg_model.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
        sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {"num_workers": cfg.data_loader_num_workers, "pin_memory": True, "sampler": sampler}
    # dataloader_kwargs = {"num_workers":0, "pin_memory": True, "sampler": sampler}

    dataloader_kwargs["batch_size"] = cfg.dataset_params.val_batch_size
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs,
                                             collate_fn=YoloXCollateFN(cfg.dataset_params.val_image_size))

    return val_loader


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""))
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)

    train_loader = get_data_loader(cfg)
    valid_loader = get_eval_loader(cfg)
    classes = COCO_DETECTION_CLASSES_LIST
    # train_loader, valid_loader, classes = None, None, None
    cfg.sg_model = SgModel(cfg.sg_model.experiment_name, cfg.model_checkpoints_location,
                           train_loader=train_loader, valid_loader=valid_loader, classes=classes,
                           multi_gpu=MultiGPUMode(cfg.multi_gpu))

    # cfg.sg_model.connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)
    cfg.sg_model.build_model(cfg.architecture, arch_params=cfg.arch_params, checkpoint_params=cfg.checkpoint_params)

    cfg.training_hyperparams.initial_lr /= 64
    cfg.training_hyperparams.initial_lr *= cfg.dataset_params.batch_size * 8
    # dvcb = DetectionVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
    #                                       freq=1,
    #                                       post_prediction_callback=YoloV5PostPredictionCallback(iou=0.65, conf=0.99),
    #                                       classes=classes,
    #                                       last_img_idx_in_batch=8)
    cfg.training_hyperparams.forward_pass_prep_fn = MultiscaleForwardPassPrepFunction(min_image_size=480, max_image_size=672)
    cfg.training_hyperparams.phase_callbacks = [YoloXTrainingStageSwitchCallback(285)]#, YoloXMultiscaleImagesCallback()]
    print(cfg.training_hyperparams.initial_lr)

    cfg.sg_model.train(training_params=cfg.training_hyperparams)


if __name__ == "__main__":
    super_gradients.init_trainer()
    main()
