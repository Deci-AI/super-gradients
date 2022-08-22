# ShelfNet LW 34 training on CoCo Segmentation Dataset:
# mIOU on CoCo Seg: ~0.66

# Since the code is training on a Subset of COCO Seg, there is an initial creation process for the "Sub-DataSet"
# this training process is optimized to enable fine-tuning on PASCAL VOC 2012 Dataset that has only 21 Classes...

# IMPORTANT: The code is optimized for a fixed initial LR since the Polynomial Loss is pretty sensitive, so we keep the
# same LR by dividing by the number of GPUs (since our code base multiplies it automatically)

# P.S. - Use the relevant training params dict if you are running on TZAG or on V100

import torch
from super_gradients.training import Trainer, MultiGPUMode
from super_gradients.training.datasets import CoCoSegmentationDatasetInterface
from super_gradients.training.sg_trainer.sg_trainer import StrictLoad
from super_gradients.training.metrics.segmentation_metrics import PixelAccuracy, IoU

model_size_str = '34'

coco_sub_classes_inclusion_tuples_list = [(0, 'background'), (5, 'airplane'), (2, 'bicycle'), (16, 'bird'),
                                          (9, 'boat'),
                                          (44, 'bottle'), (6, 'bus'), (3, 'car'), (17, 'cat'), (62, 'chair'),
                                          (21, 'cow'),
                                          (67, 'dining table'), (18, 'dog'), (19, 'horse'), (4, 'motorcycle'),
                                          (1, 'person'),
                                          (64, 'potted plant'), (20, 'sheep'), (63, 'couch'), (7, 'train'),
                                          (72, 'tv')]

coco_seg_dataset_tzag_params = {
    "batch_size": 24,
    "test_batch_size": 24,
    "dataset_dir": "/data/coco/",
    "s3_link": None,
    "img_size": 608,
    "crop_size": 512
}

coco_seg_dataset_v100_params = {
    "batch_size": 32,
    "test_batch_size": 32,
    "dataset_dir": "/home/ubuntu/data/coco/",
    "s3_link": None,
    "img_size": 608,
    "crop_size": 512
}

shelfnet_coco_training_params = {
    "max_epochs": 150, "initial_lr": 5e-3, "loss": "shelfnet_ohem_loss",
    "optimizer": "SGD", "mixed_precision": True, "lr_mode": "poly",
    "optimizer_params": {"momentum": 0.9, "weight_decay": 1e-4, "nesterov": False},
    "load_opt_params": False, "train_metrics_list": [PixelAccuracy(), IoU(21)],
    "valid_metrics_list": [PixelAccuracy(), IoU(21)],
    "loss_logging_items_names": ["Loss1/4", "Loss1/8", "Loss1/16", "Loss"], "metric_to_watch": "IoU",
    "greater_metric_to_watch_is_better": True}

shelfnet_lw_arch_params = {"num_classes": 21, "load_checkpoint": True, "strict_load": StrictLoad.ON,
                           "multi_gpu_mode": "data_parallel", "load_weights_only": True,
                           "load_backbone": True, "source_ckpt_folder_name": 'resnet' + model_size_str}

data_loader_num_workers = 8 * torch.cuda.device_count()

# BUILD THE LIGHT-WEIGHT SHELFNET ARCHITECTURE FOR TRAINING
experiment_name_prefix = 'shelfnet_lw_'
experiment_name_dataset_suffix = '_coco_seg_' + str(
    shelfnet_coco_training_params['max_epochs']) + '_epochs_train_example'

experiment_name = experiment_name_prefix + model_size_str + experiment_name_dataset_suffix

trainer = Trainer(experiment_name,
                  multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL,
                  ckpt_name='ckpt_best.pth')

coco_seg_datasaet_interface = CoCoSegmentationDatasetInterface(dataset_params=coco_seg_dataset_tzag_params,
                                                               cache_labels=False,
                                                               dataset_classes_inclusion_tuples_list=coco_sub_classes_inclusion_tuples_list)

trainer.connect_dataset_interface(coco_seg_datasaet_interface, data_loader_num_workers=data_loader_num_workers)
trainer.build_model('shelfnet' + model_size_str, arch_params=shelfnet_lw_arch_params)

print('Training ShelfNet-LW model: ' + experiment_name)
trainer.train(training_params=shelfnet_coco_training_params)
