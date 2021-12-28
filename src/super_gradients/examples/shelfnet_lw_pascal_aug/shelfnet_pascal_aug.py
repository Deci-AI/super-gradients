# TODO: REFACTOR AS YAML FILES RECIPE
import super_gradients
import torch
from super_gradients.training.datasets import PascalAUG2012SegmentationDataSetInterface
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.sg_model.sg_model import StrictLoad
from super_gradients.training.metrics.segmentation_metrics import PixelAccuracy, IoU

super_gradients.init_trainer()
pascal_aug_dataset_params = {"batch_size": 16,
                             "test_batch_size": 16,
                             "dataset_dir": "/data/pascal_voc_2012/VOCaug/dataset/",
                             "s3_link": None,
                             "img_size": 512,
                             "train_loader_drop_last": True,
                             }
shelfnet_lw_pascal_aug_training_params = {"max_epochs": 250, "initial_lr": 1e-2, "loss": "shelfnet_ohem_loss",
                                          "optimizer": "SGD", "mixed_precision": False, "lr_mode": "poly",
                                          "optimizer_params": {"momentum": 0.9, "weight_decay": 1e-4,
                                                               "nesterov": False},
                                          "load_opt_params": False, "train_metrics_list": [PixelAccuracy(), IoU(21)],
                                          "valid_metrics_list": [PixelAccuracy(), IoU(21)],
                                          "loss_logging_items_names": ["Loss1/4", "Loss1/8", "Loss1/16", "Loss"],
                                          "metric_to_watch": "IoU",
                                          "greater_metric_to_watch_is_better": True}

shelfnet_lw_arch_params = {"num_classes": 21, "load_checkpoint": True, "strict_load": StrictLoad.ON,
                           "multi_gpu_mode": MultiGPUMode.OFF, "load_weights_only": True,
                           "load_backbone": True, "source_ckpt_folder_name": 'resnet_backbones'}
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    data_loader_num_workers = 16
    shelfnet_lw_pascal_aug_training_params["initial_lr"] = shelfnet_lw_pascal_aug_training_params["initial_lr"] / 2.
else:
    # SINGLE GPU TRAINING
    data_loader_num_workers = 8
epoc_metrics_headers = {"Epoch": 0, "gpu_mem": 0.0, "Loss1/4": 0.0, "Loss1/8": 0.0, "Loss1/16": 0.0,
                        "TrainLoss": 0.0, "targets": 0, "img_size": 0}
results_titles = ['LossP', 'Loss8', 'Loss16', 'Train Loss', 'pixAcc', 'mIOU', 'Test Loss']

# SET THE *LIGHT-WEIGHT* SHELFNET ARCHITECTURE SIZE (UN-COMMENT TO TRAIN)
model_size_str = '34'
# model_size_str = '18'
# BUILD THE LIGHT-WEIGHT SHELFNET ARCHITECTURE FOR TRAINING
experiment_name_prefix = 'shelfnet_lw_'
experiment_name_dataset_suffix = '_pascal_aug_encoding_dataset_train_250_epochs_no_batchnorm_decoder'
experiment_name = experiment_name_prefix + model_size_str + experiment_name_dataset_suffix
model = SgModel(experiment_name, model_checkpoints_location='local', multi_gpu=True,
                ckpt_name='resnet' + model_size_str + '.pth',
                epoch_metric_headers=epoc_metrics_headers,
                results_titles=results_titles)

pascal_aug_datasaet_interface = PascalAUG2012SegmentationDataSetInterface(
    dataset_params=pascal_aug_dataset_params,
    cache_labels=False)
model.connect_dataset_interface(pascal_aug_datasaet_interface, data_loader_num_workers=data_loader_num_workers)
model.build_model('shelfnet' + model_size_str, arch_params=shelfnet_lw_arch_params)
print('Training ShelfNet-LW model: ' + experiment_name)
model.train(training_params=shelfnet_lw_pascal_aug_training_params)
