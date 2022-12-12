"""
Cifar10 training with SuperGradients training with the following initialized torch objects:

    DataLoaders
    Optimizers
    Networks (nn.Module)
    Schedulers
    Loss functions

Main purpose is to demonstrate training in SG with minimal abstraction and maximal flexibility
"""

from super_gradients import Trainer
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from torch.optim import ASGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from super_gradients.training.utils.callbacks import Phase, LRSchedulerCallback
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# Define any torch DataLoaders, need at least train & valid loaders
train_dataset = CIFAR10(root="data/", download=True, train=True, transform=ToTensor())
valid_dataset = CIFAR10(root="data/", download=True, train=False, transform=ToTensor())

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Define any network of type nn.Module
net = resnet18(num_classes=len(train_dataset.classes))

# Define any optimizer of type torch.optim.Optimizer (and schedulers)
lr = 2.5e-4
optimizer = ASGD(net.parameters(), lr=lr, weight_decay=0.0001)

rop_lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=10, verbose=True)
step_lr_scheduler = MultiStepLR(optimizer, milestones=[0, 150, 200], gamma=0.1)

# Define any loss function of type torch.nn.modules.loss._Loss
loss_fn = CrossEntropyLoss()

# Define phase callbacks
phase_callbacks = [
    LRSchedulerCallback(scheduler=rop_lr_scheduler, phase=Phase.VALIDATION_EPOCH_END, metric_name="Accuracy"),
    LRSchedulerCallback(scheduler=step_lr_scheduler, phase=Phase.TRAIN_EPOCH_END),
]

# Bring everything together with Trainer and start training
trainer = Trainer("Cifar10_external_objects_example")

train_params = {
    "max_epochs": 300,
    "phase_callbacks": phase_callbacks,
    "initial_lr": lr,
    "loss": loss_fn,
    "criterion_params": {},
    "optimizer": optimizer,
    "train_metrics_list": [Accuracy(), Top5()],
    "valid_metrics_list": [Accuracy(), Top5()],
    "metric_to_watch": "Accuracy",
    "greater_metric_to_watch_is_better": True,
    "lr_scheduler_step_type": "epoch",
}

trainer.train(model=net, training_params=train_params, train_loader=train_loader, valid_loader=valid_loader)
