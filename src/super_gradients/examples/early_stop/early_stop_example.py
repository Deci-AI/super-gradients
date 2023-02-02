# Cifar10 Classification Training:
# Reaches ~94.9 Accuracy after 250 Epochs
import super_gradients
from super_gradients import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models, dataloaders
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase

# Define Parameters
super_gradients.init_trainer()

early_stop_acc = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="Accuracy", mode="max", patience=3, verbose=True)
early_stop_val_loss = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="LabelSmoothingCrossEntropyLoss", mode="min", patience=3, verbose=True)

train_params = {
    "max_epochs": 250,
    "lr_updates": [100, 150, 200],
    "lr_decay_factor": 0.1,
    "lr_mode": "step",
    "lr_warmup_epochs": 0,
    "initial_lr": 0.1,
    "loss": "cross_entropy",
    "optimizer": "SGD",
    "criterion_params": {},
    "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
    "train_metrics_list": [Accuracy(), Top5()],
    "valid_metrics_list": [Accuracy(), Top5()],
    "metric_to_watch": "Accuracy",
    "greater_metric_to_watch_is_better": True,
    "phase_callbacks": [early_stop_acc, early_stop_val_loss],
}

# Define Model
trainer = Trainer("Callback_Example")

# Build Model
model = models.get(Models.RESNET18_CIFAR, num_classes=10)

trainer.train(model=model, training_params=train_params, train_loader=dataloaders.cifar10_train(), valid_loader=dataloaders.cifar10_val())
