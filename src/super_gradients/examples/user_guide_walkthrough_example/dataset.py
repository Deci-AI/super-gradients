"""
This file is used to define the Dataset used for the Training.
"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from super_gradients.training import utils as core_utils
from super_gradients.training.datasets.dataset_interfaces import DatasetInterface


class UserDataset(DatasetInterface):
    """
    The user's dataset inherits from SuperGradient's DatasetInterface and must
    contain a trainset and testset from which the the data will be loaded using.
    All augmentations, resizing and parsing must be done in this class.

     - Augmentations are defined below and will be carried out in the order they are given.
       super_gradients provides additional dataset reading tools such as ListDataset given a list of files
       corresponding to the images and labels.

    """
    def __init__(self, name="cifar10", dataset_params={}):
        super(UserDataset, self).__init__(dataset_params)
        self.dataset_name = name
        self.lib_dataset_params = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)}

        crop_size = core_utils.get_param(self.dataset_params, 'crop_size', default_val=32)

        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])

        self.trainset = datasets.CIFAR10(root=self.dataset_params.dataset_dir, train=True, download=True,
                                         transform=transform_train)

        self.testset = datasets.CIFAR10(root=self.dataset_params.dataset_dir, train=False, download=True,
                                        transform=transform_test)
