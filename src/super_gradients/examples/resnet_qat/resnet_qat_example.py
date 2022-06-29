import super_gradients
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.metrics.classification_metrics import Accuracy
from super_gradients.training.utils.quantization_utils import Int8CalibrationPreTrainingCallback, \
    PostQATConversionCallback, save_trt_engine_from_onnx_ckpt, QuantizationLevel
#
# dataset_params = {"batch_size": 64}
# dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params=dataset_params)
# model = SgModel("resnet18_qat_imagenet_15_epochs_entropy_calib_ddp",
#                 model_checkpoints_location='local',
#                 multi_gpu=MultiGPUMode.OFF)
# model.connect_dataset_interface(dataset)
# model.build_model("resnet18", checkpoint_params={"load_checkpoint": True},
#                   arch_params={"use_quant_modules": True,
#                                "quant_modules_calib_method": "entropy"})
#
# train_params = {"max_epochs": 16,
#                 "lr_mode": "cosine",
#                 "cosine_final_lr_ratio": 0.01,
#                 "initial_lr": 0.0001, "loss": "cross_entropy", "train_metrics_list": [Accuracy()],
#                 "valid_metrics_list": [Accuracy()],
#                 "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
#                 "greater_metric_to_watch_is_better": True, "average_best_models": False,
#                 "phase_callbacks": [#Int8CalibrationPreTrainingCallback(num_calib_batches=16),
#                                     PostQATConversionCallback((1, 3, 224, 224))]}
#
# model.train(training_params=train_params)
#
# import infery
# model = infery.load(model_path='/home/shay.aharon/PycharmProjects/super_gradients/checkpoints/resnet18_qat_imagenet_15_epochs_entropy_calib_ddp/ckpt_best.engine',
#                     framework_type='trt')
# benchmark_res=model.benchmark(1)
# print(benchmark_res)
#

# save_trt_engine_from_onnx_ckpt("/home/shay.aharon/PycharmProjects/super_gradients/checkpoints/resnet18_qat_imagenet_15_epochs_entropy_calib_ddp/baseline_ckpt.onnx",
#                                "/home/shay.aharon/PycharmProjects/super_gradients/checkpoints/resnet18_qat_imagenet_15_epochs_entropy_calib_ddp/baseline_ckpt.engine",
#                                QuantizationLevel.FP16)

import infery
model = infery.load(model_path='/home/shay.aharon/PycharmProjects/super_gradients/checkpoints/resnet18_qat_imagenet_15_epochs_entropy_calib_ddp/ckpt_best.engine',
                    framework_type='trt')
benchmark_res=model.benchmark(1, include_io=False)
print(benchmark_res)


# dataset_params = {"batch_size": 64}
# dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params=dataset_params)
# model = SgModel("resnet18_qat_imagenet_15_epochs_entropy_calib_ddp",
#                 model_checkpoints_location='local',
#                 multi_gpu=MultiGPUMode.OFF)
# model.connect_dataset_interface(dataset)
# model.build_model("resnet18", checkpoint_params={"pretrained_weights": "imagenet"})
# net = model.net
# dummy_input = torch.randn((1,3,224,224), device='cuda')
# torch.onnx.export(net, dummy_input, "/home/shay.aharon/PycharmProjects/super_gradients/checkpoints/resnet18_qat_imagenet_15_epochs_entropy_calib_ddp/baseline_ckpt.onnx", verbose=False, opset_version=13, enable_onnx_checker=False,
#                   do_constant_folding=True)