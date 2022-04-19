import torch
import unittest
import numpy as np
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

try:
    import tensorflow.keras as keras

    _imported_tf_failiure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warn(
        'Failed to tensorflow- only affects TestExternalDatasetInterface which uses keras keras.utils.Sequence.')
    _imported_tf_failiure = import_err

from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ExternalDatasetInterface


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=1, dim=(320, 320), n_channels=3,
                 n_classes=1000, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = np.ones(1000)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        dataset_len = 32
        return dataset_len

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        X, y = self.__data_generation(list_IDs_temp)
        return X.squeeze(axis=0), y.squeeze(axis=0)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        X = np.ones((self.batch_size, self.n_channels, *self.dim), dtype=np.float32)
        y = np.ones((self.batch_size, 1), dtype=np.float32)
        return X, y


class TestExternalDatasetInterface(unittest.TestCase):

    def setUp(self):
        params = {'dim': (256, 256),
                  'batch_size': 1,
                  'n_classes': 1000,
                  'n_channels': 3,
                  'shuffle': True}
        training_generator = DataGenerator(**params)
        testing_generator = DataGenerator(**params)
        external_num_classes = 1000
        external_dataset_params = {'batch_size': 16,
                                   "val_batch_size": 16}
        self.dim = params['dim'][0]
        self.n_channels = params['n_channels']
        self.batch_size = external_dataset_params['batch_size']
        self.val_batch_size = external_dataset_params['val_batch_size']
        self.test_external_dataset_interface = ExternalDatasetInterface(train_loader=training_generator,
                                                                        val_loader=testing_generator,
                                                                        num_classes=external_num_classes,
                                                                        dataset_params=external_dataset_params)

    def test_get_data_loaders(self):
        train_loader, val_loader, _, num_classes = self.test_external_dataset_interface.get_data_loaders()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            self.assertListEqual([self.batch_size, self.n_channels, self.dim, self.dim], list(inputs.shape))
            self.assertListEqual([self.batch_size, 1], list(targets.shape))
            self.assertEqual(torch.Tensor, type(inputs))
            self.assertEqual(torch.Tensor, type(targets))
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            self.assertListEqual([self.val_batch_size, self.n_channels, self.dim, self.dim], list(inputs.shape))
            self.assertListEqual([self.val_batch_size, 1], list(targets.shape))
            self.assertEqual(torch.Tensor, type(inputs))
            self.assertEqual(torch.Tensor, type(targets))


if __name__ == '__main__':
    unittest.main()
