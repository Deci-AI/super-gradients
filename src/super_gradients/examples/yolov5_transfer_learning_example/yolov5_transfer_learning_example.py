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
