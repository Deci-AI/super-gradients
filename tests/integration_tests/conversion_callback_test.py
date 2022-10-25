import unittest
from enum import Enum
import re

from super_gradients.training import models

from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import segmentation_test_dataloader, classification_test_dataloader
from super_gradients.training.utils.callbacks import ModelConversionCheckCallback
from super_gradients.training.metrics import Accuracy, Top5, IoU
from super_gradients.training.losses.stdc_loss import STDCLoss
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss

from deci_lab_client.models import ModelMetadata, HardwareType, FrameworkType

checkpoint_dir = "/Users/daniel/Documents/LALA"


class Task(Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


def generate_model_metadata(architecture: str, task: Task):
    model_name = f"{architecture}_for_testing"
    return ModelMetadata(
        name=model_name,
        primary_batch_size=1,
        architecture=architecture.title(),
        framework=FrameworkType.PYTORCH,
        dl_task=task.value,
        input_dimensions=(3, 320, 320),
        primary_hardware=HardwareType.K80,
        dataset_name="ImageNet",
        description=f"{model_name} deci.ai Test",
        tags=["imagenet", model_name],
    )


CLASSIFICATION = ["efficientnet_b0", "regnetY200", "regnetY400", "regnetY600", "regnetY800", "mobilenet_v3_large"]
SEMANTIC_SEGMENTATION = ["ddrnet_23", "stdc1_seg", "stdc2_seg", "regseg48"]


# TODO: ADD YOLOX ARCHITECTURES AND TESTS


class ConversionCallbackTest(unittest.TestCase):
    def test_classification_architectures(self):
        for architecture in CLASSIFICATION:
            model_meta_data = generate_model_metadata(architecture=architecture, task=Task.CLASSIFICATION)
            phase_callbacks = [ModelConversionCheckCallback(model_meta_data=model_meta_data, opset_version=11)]
            train_params = {
                "max_epochs": 2,
                "lr_updates": [1],
                "lr_decay_factor": 0.1,
                "lr_mode": "step",
                "lr_warmup_epochs": 0,
                "initial_lr": 0.1,
                "loss": "cross_entropy",
                "optimizer": "SGD",
                "criterion_params": {},
                "train_metrics_list": [Accuracy(), Top5()],
                "valid_metrics_list": [Accuracy(), Top5()],
                "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True,
                "phase_callbacks": phase_callbacks,
            }

            trainer = Trainer(f"{architecture}_example", ckpt_root_dir=checkpoint_dir)
            model = models.get(architecture=architecture, arch_params={"use_aux_heads": True, "aux_head": True})
            try:
                trainer.train(
                    model=model, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
                )
            except Exception as e:
                self.fail(f"Model training didn't succeed due to {e}")
            else:
                self.assertTrue(True)

    def test_segmentation_architectures(self):
        def get_architecture_custom_config(architecture_name: str):
            if re.search(r"ddrnet", architecture_name):
                return {
                    "loss": DDRNetLoss(num_pixels_exclude_ignored=False),
                }
            elif re.search(r"stdc", architecture_name):
                return {
                    "loss": STDCLoss(num_classes=5),
                }
            elif re.search(r"regseg", architecture_name):
                return {
                    "loss": "cross_entropy",
                }
            else:
                raise Exception("You tried to run a conversion test on an unknown architecture")

        for architecture in SEMANTIC_SEGMENTATION:
            model_meta_data = generate_model_metadata(architecture=architecture, task=Task.SEMANTIC_SEGMENTATION)
            trainer = Trainer(f"{architecture}_example", ckpt_root_dir=checkpoint_dir)
            model = models.get(model_name=architecture, arch_params={"use_aux_heads": True, "aux_head": True})

            phase_callbacks = [
                ModelConversionCheckCallback(model_meta_data=model_meta_data, opset_version=11, rtol=1, atol=1),
            ]

            train_params = {
                "max_epochs": 3,
                "initial_lr": 1e-2,
                "lr_mode": "poly",
                "ema": True,  # unlike the paper (not specified in paper)
                "optimizer": "SGD",
                "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
                "load_opt_params": False,
                "train_metrics_list": [IoU(5)],
                "valid_metrics_list": [IoU(5)],
                "metric_to_watch": "IoU",
                "greater_metric_to_watch_is_better": True,
                "phase_callbacks": phase_callbacks,
            }
            custom_config = get_architecture_custom_config(architecture_name=architecture)
            train_params.update(custom_config)

            try:
                trainer.train(
                    model=model,
                    training_params=train_params,
                    train_loader=segmentation_test_dataloader(image_size=512),
                    valid_loader=segmentation_test_dataloader(image_size=512),
                )
            except Exception as e:
                self.fail(f"Model training didn't succeed for {architecture} due to {e}")
            else:
                self.assertTrue(True)
