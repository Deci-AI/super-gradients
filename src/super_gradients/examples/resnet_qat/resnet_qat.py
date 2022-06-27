import datetime
import os
import sys
import time

import torch
import torch.utils.data
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface
import torchvision
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor

from pytorch_quantization import quant_modules
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.utils.quantization_utils import collect_stats, compute_amax, export_onnx

quant_modules.initialize()
quant_desc = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)


net = torchvision.models.resnet50(pretrained=True)
dataset_params = {"batch_size": 64}
dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params=dataset_params)
model = SgModel("resnet50_imagenet",
                model_checkpoints_location='local',
                multi_gpu=MultiGPUMode.OFF)

model.connect_dataset_interface(dataset)
model.build_model(net)

data_loader = model.valid_loader
# model.test(test_loader=dataset.val_loader, test_metrics_list=[Accuracy(), Top5()])
with torch.no_grad():
    collect_stats(model.net, data_loader, num_batches=2)
    compute_amax(model.net, method="percentile", percentile=99.99)
# model.test(test_loader=dataset.val_loader, test_metrics_list=[Accuracy(), Top5()])

# FINE TUNE CALIBRATED MODEL
train_params = {"max_epochs": 1, "lr_mode": "step", "lr_updates": [2], "lr_decay_factor": 0.1,
                "initial_lr": 0.0001, "loss": "cross_entropy", "train_metrics_list": [Accuracy(), Top5()],
                "valid_metrics_list": [Accuracy(), Top5()],
                "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True}
model.train(training_params=train_params)
export_onnx(model=model.net,onnx_filename="resnet50_imagenet_qat.onnx",batch_onnx=1,per_channel_quantization=False)