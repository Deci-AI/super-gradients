from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val

import argparse

# In case you are running:
# W&B logger:
# -> pip install wandb
# -> wandb login
# -> python logger_examples.py -p my-first-project -l wandb_sg_logger

# ClearML logger:
# -> pip install clearml
# -> clearml-init
# -> python logger_examples.py -p my-first-project -l clearml_sg_logger

# Dagshub logger:
# -> pip install dagshub mlflow
# -> python logger_examples.py -p my-first-project -l dagshub_sg_logger


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project_name", default="demo-wandb-cifar")
parser.add_argument("-l", "--logger", default="wandb_sg_logger", choices=["wandb_sg_logger", "clearml_sg_logger", "dagshub_sg_logger"])

if __name__ == "__main__":
    args = parser.parse_args()

    trainer = Trainer(experiment_name=f"demo-{args.logger}")
    model = models.get(Models.RESNET18, num_classes=10)

    training_params = {
        "max_epochs": 20,
        "lr_updates": [5, 10, 15],
        "lr_decay_factor": 0.1,
        "lr_mode": "StepLRScheduler",
        "initial_lr": 0.1,
        "loss": "CrossEntropyLoss",
        "optimizer": "SGD",
        "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
        "train_metrics_list": [Accuracy(), Top5()],
        "valid_metrics_list": [Accuracy(), Top5()],
        "metric_to_watch": "Accuracy",
        "greater_metric_to_watch_is_better": True,
        "sg_logger": args.logger,
        "sg_logger_params": {
            "project_name": args.project_name,
            "save_checkpoints_remote": False,
            "save_tensorboard_remote": False,
            "save_logs_remote": False,
        },
        "max_train_batches": 24,
        "max_valid_batches": 24,
    }

    trainer.train(model=model, training_params=training_params, train_loader=cifar10_train(), valid_loader=cifar10_val())
