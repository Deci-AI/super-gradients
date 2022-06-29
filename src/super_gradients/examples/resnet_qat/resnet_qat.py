import torch.utils.data
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.metrics.classification_metrics import Accuracy
from super_gradients.training.utils.quantization_utils import collect_stats, compute_amax
import super_gradients

super_gradients.init_trainer()

quant_modules.initialize()
quant_desc = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)


dataset_params = {"batch_size": 32}
dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params=dataset_params)


model = SgModel("resnet18_qat_imagenet",
                model_checkpoints_location='local',
                multi_gpu=MultiGPUMode.OFF)

model.connect_dataset_interface(dataset)

model.build_model("resnet18", checkpoint_params={"pretrained_weights": "imagenet"})
test_data_loader = model.valid_loader

with torch.no_grad():
    collect_stats(model.net, test_data_loader, num_batches=2)
    compute_amax(model.net, method="percentile", percentile=99.99)

print("Testing pretrained resnet18 after calibration and before QAT, original model accuracy is 0.706")

train_params = {"max_epochs": 2,
                "lr_mode": "step",
                "lr_updates": [2], "lr_decay_factor": 0.1,
                "initial_lr": 0.0001, "loss": "cross_entropy", "train_metrics_list": [Accuracy()],
                "valid_metrics_list": [Accuracy()],
                "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True, "average_best_models": False}
model.train(training_params=train_params)

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')  # TODO: switch input dims by model
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model.net, dummy_input, "sg_torchvision_resnet50_qat_1_epoch.onnx", verbose=False, opset_version=13,
                  enable_onnx_checker=False, do_constant_folding=True)

# FINE TUNE CALIBRATED MODEL
