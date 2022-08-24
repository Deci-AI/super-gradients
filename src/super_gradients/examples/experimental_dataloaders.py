from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.dataloaders.dataloader_factory import imagenet_train, imagenet_val
import super_gradients

from hydra import initialize, compose
import hydra
from hydra.core.global_hydra import GlobalHydra
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN

super_gradients.init_trainer()
sm = SgModel("sanity_checkdl", multi_gpu=MultiGPUMode.OFF)
dltrain = imagenet_train()
dlval = imagenet_val()
print(dlval)


for x, y in dltrain:
    pass
# from super_gradients.training.datasets.dataset_interfaces import ImageNetDatasetInterface
#
# di = ImageNetDatasetInterface(
#     {
#         "batch_size": 64,
#         "val_batch_size": 200,
#         "dataset_dir": "/data/Imagenet",
#         "traindir": "train",
#         "valdir": "val",
#         "img_mean": [0.485, 0.456, 0.406],
#         "img_std": [0.229, 0.224, 0.225],
#         "crop_size": 224,
#         "resize_size": 256,
#         "color_jitter": 0.0,
#         "imagenet_pca_aug": 0.0,
#         "train_interpolation": "default",
#         "rand_augment_config_string": None,
#         "random_erase_prob": 0.0,
#         "aug_repeat_count": 0
#     }
# )
#
# print(1)