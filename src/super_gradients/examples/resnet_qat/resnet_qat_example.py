from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.metrics.classification_metrics import Accuracy
from super_gradients.training.utils.quantization_utils import Int8CalibrationPreTrainingCallback, \
    PostQATConversionCallback

dataset_params = {"batch_size": 32}
dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params=dataset_params)
model = SgModel("resnet18_qat_imagenet",
                model_checkpoints_location='local',
                multi_gpu=MultiGPUMode.OFF)
model.connect_dataset_interface(dataset)
model.build_model("resnet18", checkpoint_params={"pretrained_weights": "imagenet"},
                  arch_params={"use_quant_modules": True})

train_params = {"max_epochs": 1,
                "lr_mode": "step",
                "lr_updates": [2], "lr_decay_factor": 0.1,
                "initial_lr": 0.0001, "loss": "cross_entropy", "train_metrics_list": [Accuracy()],
                "valid_metrics_list": [Accuracy()],
                "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True, "average_best_models": False,
                "phase_callbacks": [Int8CalibrationPreTrainingCallback(), PostQATConversionCallback((1, 3, 224, 224))]}

model.train(training_params=train_params)
