# Cifar10 Classification Training:
# Reaches ~94.9 Accuracy after 250 Epochs
import super_gradients
from super_gradients import SgModel
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import Cifar10DatasetInterface
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase
# Define Parameters
super_gradients.init_trainer()

early_stop_acc = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="Accuracy", mode="max", patience=3, verbose=True)
early_stop_val_loss = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="Loss", mode="min", patience=3, verbose=True)

train_params = {"max_epochs": 250, "lr_updates": [100, 150, 200], "lr_decay_factor": 0.1, "lr_mode": "step",
                "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": "SGD",
                "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True, "phase_callbacks": [early_stop_acc, early_stop_val_loss]}

# Define Model
model = SgModel("Callback_Example")

# Connect Dataset
dataset = Cifar10DatasetInterface()
model.connect_dataset_interface(dataset, data_loader_num_workers=8)

# Build Model
model.build_model("resnet18_cifar", load_checkpoint=False)
model.train(training_params=train_params)
