"""
QAT example for Resnet18

The purpose of this example is to demonstrate the usage of QAT in super_gradients.

Behind the scenes, when passing enable_qat=True, a callback for QAT will be added.

Once triggered, the following will happen:
- The model will be rebuilt with quantized nn.modules.
- The pretrained imagenet weights will be loaded to it.
- We perform calibration with 2 batches from our training set (1024 samples = 8 gpus X 128 samples_per_batch).
- We evaluate the calibrated model (accuracy is logged under calibrated_model_accuracy).
- The calibrated checkpoint prior to QAT is saved under ckpt_calibrated_{calibration_method}.pth.
- We fine tune the calibrated model for 1 epoch.

Finally, once training is over- we trigger a pos-training callback that will export the ONNX files.

"""

from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface
from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.metrics.classification_metrics import Accuracy

import super_gradients
from super_gradients.training.utils.quantization_utils import PostQATConversionCallback

super_gradients.init_trainer()

dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params={"batch_size": 128})
model = SgModel("resnet18_qat_example",
                model_checkpoints_location='local',
                multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL)

model.connect_dataset_interface(dataset)
model.build_model("resnet18", checkpoint_params={"pretrained_weights": "imagenet"})

train_params = {"max_epochs": 1,
                "lr_mode": "step",
                "optimizer": "SGD",
                "lr_updates": [],
                "lr_decay_factor": 0.1,
                "initial_lr": 0.001, "loss": "cross_entropy",
                "train_metrics_list": [Accuracy()],
                "valid_metrics_list": [Accuracy()],
                "loss_logging_items_names": ["Loss"],
                "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True,
                "average_best_models": False,
                "enable_qat": True,
                "qat_params": {
                    "start_epoch": 0,  # first epoch for quantization aware training.
                    "quant_modules_calib_method": "percentile",
                    # statistics method for amax computation (one of [percentile, mse, entropy, max]).
                    "calibrate": True,  # whether to perform calibration.
                    "num_calib_batches": 2,  # number of batches to collect the statistics from.
                    "percentile": 99.99  # percentile value to use when SgModel,
                },
                "phase_callbacks": [PostQATConversionCallback(dummy_input_size=(1, 3, 224, 224))]
                }

model.train(training_params=train_params)
