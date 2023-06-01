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
from deci_platform_client.models import QuantizationLevel, HardwareType


def main(architecture_name: str):
    # Empty on purpose so that it can be fit to the trainer use case
    checkpoint_dir = ""

    # You can also set your token as environment variable using the commandline or your IDE.
    os.environ["DECI_CLIENT_ID"] = "YOUR_CLIENT_ID_HERE"
    os.environ["DECI_CLIENT_SECRET"] = "YOUR_SECRET_KEY_HERE"
    trainer = Trainer(
        f"lab_optimization_{architecture_name}_example",
        model_checkpoints_location="local",
        ckpt_root_dir=checkpoint_dir,
    )

    model = models.get(architecture=architecture_name, arch_params={"use_aux_heads": True})

    # CREATE META-DATA, AND OPTIMIZATION REQUEST FORM FOR DECI PLATFORM POST TRAINING CALLBACK
    model_name = f"{architecture_name}_for_deci_lab_export_example"

    # IT IS ALSO RECOMMENDED TO USE A PRE TRAINING MODEL CONVERSION CHECK CALLBACK, SO THAT ANY CONVERSION
    # ERRORS WON'T APPEAR FOR THE FIRST TIME ONLY AT THE END OF TRAINING:

    phase_callbacks = [
        ModelConversionCheckCallback(
            model_name=model_name,
            input_dimensions=(3, 320, 320),
            primary_batch_size=1,
            opset_version=11,
        ),
        DeciLabUploadCallback(
            model_name=model_name,
            input_dimensions=(3, 320, 320),
            target_hardware_types=[HardwareType.T4],
            target_batch_size=1,
            target_quantization_level=QuantizationLevel.FP16,
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
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=classification_test_dataloader(),
        valid_loader=classification_test_dataloader(),
    )


if __name__ == "__main__":
    main(architecture_name="efficientnet_b0")
