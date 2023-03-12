"""
Deci-lab model export example.

The main purpose of this code is to demonstrate how to upload the model to the platform, optimize and download it
 after training is complete, using DeciPlatformCallback.
"""
import os
from super_gradients.training import models

from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils.callbacks import DeciLabUploadCallback, ModelConversionCheckCallback
from deci_lab_client.models import (
    Metric,
    QuantizationLevel,
    ModelMetadata,
    OptimizationRequestForm,
    HardwareType,
    FrameworkType,
)


def main(architecture_name: str):
    # Empty on purpose so that it can be fit to the trainer use case
    checkpoint_dir = ""

    os.environ["DECI_PLATFORM_TOKEN"] = "YOUR_API_TOKEN_HERE"  # You can also set your token as environment variable using the commandline or your IDE.
    trainer = Trainer(
        f"lab_optimization_{architecture_name}_example",
        model_checkpoints_location="local",
        ckpt_root_dir=checkpoint_dir,
    )

    model = models.get(architecture=architecture_name, arch_params={"use_aux_heads": True})

    # CREATE META-DATA, AND OPTIMIZATION REQUEST FORM FOR DECI PLATFORM POST TRAINING CALLBACK
    model_name = f"{architecture_name}_for_deci_lab_export_example"
    model_meta_data = ModelMetadata(
        name=model_name,
        primary_batch_size=1,
        architecture=architecture_name.title(),
        framework=FrameworkType.PYTORCH,
        dl_task="classification",
        input_dimensions=(3, 320, 320),
        primary_hardware=HardwareType.K80,
        dataset_name="ImageNet",
        description=f"{architecture_name} deci.ai Test",
        tags=["imagenet", architecture_name],
    )

    optimization_request_form = OptimizationRequestForm(
        target_hardware=HardwareType.T4,
        target_batch_size=1,
        target_metric=Metric.LATENCY,
        optimize_model_size=True,
        quantization_level=QuantizationLevel.FP16,
        optimize_autonac=True,
    )

    # IT IS ALSO RECOMMENDED TO USE A PRE TRAINING MODEL CONVERSION CHECK CALLBACK, SO THAT ANY CONVERSION
    # ERRORS WON'T APPEAR FOR THE FIRST TIME ONLY AT THE END OF TRAINING:

    phase_callbacks = [
        ModelConversionCheckCallback(model_meta_data=model_meta_data, opset_version=11),
        DeciLabUploadCallback(
            model_meta_data=model_meta_data,
            optimization_request_form=optimization_request_form,
            opset_version=11,
        ),
    ]

    # DEFINE TRAINING PARAMETERS
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

    # RUN TRAINING. ONCE ALL EPOCHS ARE DONE THE OPTIMIZED MODEL FILE WILL BE LOCATED IN THE EXPERIMENT'S
    # CHECKPOINT DIRECTORY
    trainer.train(model=model, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())


if __name__ == "__main__":
    main(architecture_name="efficientnet_b0")
