# Yolo v5 Detection training on CoCo2017 Dataset:
# Yolo v5s train on 320x320 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~28.4
# Yolo v5s train in 640x640 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~29.1

# Yolo v5 Detection training on CoCo2014 Dataset:
# Yolo v5s train on 320x320 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~28.77

# batch size may need to change depending on model size and GPU (2080Ti, V100)
# The code is optimized for running with a Mini-Batch of 64 examples... So depending on the amount of GPUs,
# you should change the "batch_accumulate" param in the training_params dict to be batch_size * gpu_num * batch_accumulate = 64.

from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch
import os
from zipfile import ZipFile

# names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#
# def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
#     # Multi-threaded file download and unzip function, used in data.yaml for autodownload
#     def download_one(url, dir):
#         # Download 1 file
#         f = dir / Path(url).name  # filename
#         if Path(url).is_file():  # exists in current path
#             Path(url).rename(f)  # move to dir
#         elif not f.exists():
#             print(f'Downloading {url} to {f}...')
#             if curl:
#                 os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
#             else:
#                 torch.hub.download_url_to_file(url, f, progress=True)  # torch download
#         if unzip and f.suffix in ('.zip', '.gz'):
#             print(f'Unzipping {f}...')
#             if f.suffix == '.zip':
#                 ZipFile(f).extractall(path=dir)  # unzip
#             elif f.suffix == '.gz':
#                 os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
#             if delete:
#                 f.unlink()  # remove zip
#
#     dir = Path(dir)
#     dir.mkdir(parents=True, exist_ok=True)  # make directory
#     for u in [url] if isinstance(url, (str, Path)) else url:
#         download_one(u, dir)
#
# def convert_label(path, lb_path, year, image_id):
#     def convert_box(size, box):
#         dw, dh = 1. / size[0], 1. / size[1]
#         x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
#         return x * dw, y * dh, w * dw, h * dh
#
#     in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
#     out_file = open(lb_path, 'w')
#     tree = ET.parse(in_file)
#     root = tree.getroot()
#     size = root.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)
#     for obj in root.iter('object'):
#         cls = obj.find('name').text
#         if cls in names and not int(obj.find('difficult').text) == 1:
#             xmlbox = obj.find('bndbox')
#             bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
#             cls_id = names.index(cls)  # class id
#             out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')
#
#
# # Download
# dir = Path("/home/shay.aharon/data/pascal_unified_coco_format")  # dataset root dir
# url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
# urls = [url + 'VOCtrainval_06-Nov-2007.zip',  # 446MB, 5012 images
#         url + 'VOCtest_06-Nov-2007.zip',  # 438MB, 4953 images
#         url + 'VOCtrainval_11-May-2012.zip']  # 1.95GB, 17126 images
# download(urls, dir=dir / 'images', delete=False)
# # Convert
# path = dir / f'images/VOCdevkit'
# for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
#     imgs_path = dir / 'images' / f'{image_set}{year}'
#     lbs_path = dir / 'labels' / f'{image_set}{year}'
#     imgs_path.mkdir(exist_ok=True, parents=True)
#     lbs_path.mkdir(exist_ok=True, parents=True)
#     image_ids = open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt').read().strip().split()
#     for id in tqdm(image_ids, desc=f'{image_set}{year}'):
#         f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
#         lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
#         f.rename(imgs_path / f.name)  # move image
#         convert_label(path, lb_path, year, id)  # convert labels to YOLO format
import super_gradients
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import \
    PascalVOCUnifiedDetectionDataSetInterface
from super_gradients.training.models.detection_models.yolov5 import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.utils.detection_utils import Anchors

super_gradients.init_trainer()

distributed = super_gradients.is_distributed()

dataset_params = {"batch_size": 48,
                  "val_batch_size": 48,
                  "train_image_size": 512,
                  "val_image_size": 512,
                  "val_collate_fn": base_detection_collate_fn,
                  "train_collate_fn": base_detection_collate_fn,
                  "train_sample_loading_method": "mosaic",
                  "val_sample_loading_method": "default",
                  "dataset_hyper_param": {
                      "hsv_h": 0.0138,  # IMAGE HSV-Hue AUGMENTATION (fraction)
                      "hsv_s": 0.664,  # IMAGE HSV-Saturation AUGMENTATION (fraction)
                      "hsv_v": 0.464,  # IMAGE HSV-Value AUGMENTATION (fraction)
                      "degrees": 0.373,  # IMAGE ROTATION (+/- deg)
                      "translate": 0.245,  # IMAGE TRANSLATION (+/- fraction)
                      "scale": 0.898,  # IMAGE SCALE (+/- gain)
                      "shear": 0.602,
                      "mixup": 0.243
                  }
                  }


model = SgModel("yolov5m_pascal_finetune_augment_fix",
                multi_gpu=MultiGPUMode.OFF,
                post_prediction_callback=YoloV5PostPredictionCallback())

dataset_interface = PascalVOCUnifiedDetectionDataSetInterface(dataset_params=dataset_params, cache_labels=True, cache_images=True)
model.connect_dataset_interface(dataset_interface, data_loader_num_workers=8)
model.build_model("yolo_v5m", arch_params={"pretrained_weights": "coco"})

post_prediction_callback = YoloV5PostPredictionCallback()

network = model.net
network = network.module if hasattr(network, 'module') else network
num_levels = network._head._modules_list[-1].detection_layers_num
train_image_size = dataset_params["train_image_size"]

num_branches_norm = 3. / num_levels
num_classes_norm = len(model.classes) / 80.
image_size_norm = train_image_size / 640.

training_params = {"max_epochs": 50,
                   "lr_mode": "cosine",
                   "initial_lr": 0.0032,
                   "cosine_final_lr_ratio": 0.12,
                   "lr_warmup_epochs": 2,
                   "batch_accumulate": 1,
                   "warmup_bias_lr": 0.05,
                   "loss": "yolo_v5_loss",
                   "criterion_params": {"anchors": Anchors(
                       anchors_list=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                                     [116, 90, 156, 198, 373, 326]], strides=[8, 16, 32]),
                       "box_loss_gain": 0.0296 * num_branches_norm,
                       "cls_loss_gain": 0.243 * num_classes_norm * num_branches_norm,
                       "cls_pos_weight": 0.631,
                       "obj_loss_gain": 0.301 * image_size_norm ** 2 * num_branches_norm,
                       "obj_pos_weight": 0.911,
                       "anchor_t": 2.91},
                   "optimizer": "SGD",
                   "warmup_momentum": 0.5,
                   "optimizer_params": {"momentum": 0.843,
                                        "weight_decay": 0.00036,
                                        "nesterov": True},
                   "mixed_precision": False,
                   "ema": True,
                   "train_metrics_list": [],
                   "valid_metrics_list": [DetectionMetrics(post_prediction_callback=post_prediction_callback,
                                                           num_cls=len(
                                                               dataset_interface.classes))],
                   "loss_logging_items_names": ["GIoU", "obj", "cls", "Loss"],
                   "metric_to_watch": "mAP@0.50:0.95",
                   "greater_metric_to_watch_is_better": True,
                   "warmup_mode": "yolov5_warmup"}

model.train(training_params=training_params)
