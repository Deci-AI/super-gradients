import torch
from super_gradients.training.datasets.dataset_interfaces import DatasetInterface
from super_gradients.training.sg_trainer import Trainer
from torchvision.models import resnet18
import numpy as np


class TestDatasetInterface(DatasetInterface):
    def __init__(self, dataset_params={}, image_size=32, batch_size=5):
        super(TestDatasetInterface, self).__init__(dataset_params)
        self.trainset = torch.utils.data.TensorDataset(torch.Tensor(np.zeros((batch_size, 3, image_size, image_size))),
                                                       torch.LongTensor(np.zeros((batch_size))))
        self.testset = self.trainset

    def get_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None, test_batch_size=None,
                         distributed_sampler=False):
        self.trainset.classes = [0, 1, 2, 3, 4]
        return super().get_data_loaders(batch_size_factor=batch_size_factor,
                                        num_workers=num_workers,
                                        train_batch_size=train_batch_size,
                                        test_batch_size=test_batch_size,
                                        distributed_sampler=distributed_sampler)


# ------------------ Loading The Model From Model.py----------------
arch_params = {'num_classes': 1000}
model = resnet18()
trainer = Trainer('Client_model_training',
                  model_checkpoints_location='local', device='cpu')
# if a torch.nn.Module is provided when building the model, the model will be integrated into deci model class
trainer.build_model(model, arch_params=arch_params)
# ------------------ Loading The Dataset From Dataset.py----------------
dataset = TestDatasetInterface()
trainer.connect_dataset_interface(dataset)
# ------------------ Loading The Loss From Loss.py -----------------
loss = 'cross_entropy'
# ------------------ Training -----------------
train_params = {"max_epochs": 100,
                "lr_mode": "step",
                "lr_updates": [30, 60, 90, 100],
                "lr_decay_factor": 0.1,
                "initial_lr": 0.025, "loss": loss}
trainer.train(train_params)
