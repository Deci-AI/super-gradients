
"""
Main purpose is to demonstrate the use of initialized optimizers and lr_Schedulers in training.
"""
import super_gradients

from super_gradients import SgModel
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import Cifar10DatasetInterface
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training import MultiGPUMode
from torch.optim import ASGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from super_gradients.training.models import ResNet18Cifar
from super_gradients.training.utils.callbacks import Phase, LRSchedulerCallback
from super_gradients.training import utils

super_gradients.init_trainer()
lr = 2.5e-4
net = ResNet18Cifar(arch_params=utils.HpmStruct(**{"num_classes": 10}))

# Define optimizer and schedulers
optimizer = ASGD(net.parameters(), lr=lr, weight_decay=0.0001)

rop_lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=10, verbose=True)
step_lr_scheduler = MultiStepLR(optimizer, milestones=[0, 150, 200], gamma=0.1)

# Learning rate will be decreased after validation accuracy did not increase for 10 epochs, or at the specified
# milestones. Notice how the callback for reduce on plateau scheduler is been called on Phase.VALIDATION_EPOCH_END
# which causes it to take the accuracy value from the validation metrics.

phase_callbacks = [LRSchedulerCallback(scheduler=rop_lr_scheduler, phase=Phase.VALIDATION_EPOCH_END, metric_name="Accuracy"),
                   LRSchedulerCallback(scheduler=step_lr_scheduler, phase=Phase.TRAIN_EPOCH_END)]

# Define Model
model = SgModel("Cifar10_external_objects_example", multi_gpu=MultiGPUMode.OFF)

# Connect Dataset
dataset = Cifar10DatasetInterface(dataset_params={"batch_size": 64})
model.connect_dataset_interface(dataset, data_loader_num_workers=8)

# Build Model
model.build_model(net, load_checkpoint=False)

train_params = {"max_epochs": 300,
                "phase_callbacks": phase_callbacks,
                "initial_lr": lr,
                "loss": "cross_entropy",
                "criterion_params": {},
                'optimizer': optimizer,
                "train_metrics_list": [Accuracy(), Top5()],
                "valid_metrics_list": [Accuracy(), Top5()],
                "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True,
                "lr_scheduler_step_type": "epoch"}

model.train(training_params=train_params)
