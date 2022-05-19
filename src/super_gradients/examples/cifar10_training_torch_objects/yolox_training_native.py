"""
Cifar10 training with SuperGradients training with the following initialized torch objects:

    DataLoaders
    Optimizers
    Networks (nn.Module)
    Schedulers
    Loss functions

Main purpose is to demonstrate training in SG with minimal abstraction and maximal flexibility
"""
import torch
from omegaconf import DictConfig
import hydra
import pkg_resources
from torch.utils.data import BatchSampler, DataLoader

from super_gradients import SgModel
import super_gradients
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training import MultiGPUMode
from super_gradients.training.datasets.detection_datasets.coco_detection_yolox import COCODataset, MosaicDetection
from super_gradients.training.datasets.samplers.infinite_sampler import InfiniteSampler
from super_gradients.training.transforms.yolox_transforms import TrainTransform, ValTransform
from super_gradients.training.utils.callbacks import YoloXTrainingStageSwitchCallback

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.datasets_utils import MultiscaleForwardPassPrepFunction, worker_init_reset_seed, \
    ComposedCollateFunction
from super_gradients.training.utils.detection_utils import YoloXCollateFN
from super_gradients.training.utils.distributed_training_utils import get_local_rank, wait_for_the_master

from loguru import logger

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
            cache=cache_img
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
@logger.catch
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
    cfg.training_hyperparams.forward_pass_prep_fn = MultiscaleForwardPassPrepFunction()
    cfg.training_hyperparams.phase_callbacks = [YoloXTrainingStageSwitchCallback(285)]#, YoloXMultiscaleImagesCallback()]
    print(cfg.training_hyperparams.initial_lr)

    cfg.sg_model.train(training_params=cfg.training_hyperparams)


if __name__ == "__main__":
    super_gradients.init_trainer()
    main()


