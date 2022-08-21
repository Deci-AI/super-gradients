import super_gradients
import torch
import unittest
import numpy as np
from PIL import Image
import tensorflow.keras as keras
from super_gradients.training import MultiGPUMode
from super_gradients.training import Trainer
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ExternalDatasetInterface, \
    ImageNetDatasetInterface
from super_gradients.training.metrics import Accuracy, Top5


class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, batch_size=1, dims=(320, 320), n_channels=3,
                 n_classes=1000, shuffle=True):
        self.dims = dims
        self.batch_size = batch_size
        self.samples = samples
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Fraction of dataset to be used - for faster testing
        fraction_of_dataset = 0.01
        return int(np.floor(len(self.samples) / self.batch_size) * fraction_of_dataset)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.samples[k] for k in indices]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dims, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            image = Image.open(ID[0])
            image = image.resize((self.dims))
            rgb_image = Image.new("RGB", image.size)
            rgb_image.paste(image)
            X[i, ] = np.array(rgb_image)
            y[i] = ID[1]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def create_imagenet_dataset():
    dataset_params = {"batch_size": 1}
    dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params=dataset_params)
    return dataset


class TransposeCollateFn(object):

    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, batch):
        new_inputs = []
        new_targets = []
        for img in batch:
            squeezed_input = img[0].squeeze(axis=0)
            transposed_data = np.transpose(squeezed_input, self.new_shape)
            new_inputs.append(torch.from_numpy(transposed_data))
            argmax_target = np.argmax(img[1], 1)
            new_targets.append(torch.from_numpy(argmax_target))
        return torch.stack(new_inputs, 0), torch.cat(new_targets, 0)


class TestExternalDatasetInterface(unittest.TestCase):

    def setUp(self):
        super_gradients.init_trainer()
        dataset = create_imagenet_dataset()
        data_samples_train = dataset.trainset.samples
        data_samples_val = dataset.valset.samples
        # batch size: 1 is only for the creation of the external keras loader
        self.keras_params = {'dims': (256, 256),
                             'batch_size': 1,
                             'n_classes': 1000,
                             'n_channels': 3,
                             'shuffle': True}
        training_generator = DataGenerator(data_samples_train, **self.keras_params)
        testing_generator = DataGenerator(data_samples_val, **self.keras_params)
        external_num_classes = 1000
        collate_fn = TransposeCollateFn((2, 0, 1))
        self.external_dataset_params = {'batch_size': 16,
                                        'test_batch_size': 16,
                                        'train_collate_fn': collate_fn,
                                        'val_collate_fn': collate_fn}
        self.test_external_dataset_interface = ExternalDatasetInterface(train_loader=training_generator,
                                                                        val_loader=testing_generator,
                                                                        num_classes=external_num_classes,
                                                                        dataset_params=self.external_dataset_params)

    def test_transpose_collate_fn(self):
        collate_fn = TransposeCollateFn((2, 0, 1))
        dims = self.keras_params['dims']
        n_channels = self.keras_params['n_channels']
        batch_size = self.external_dataset_params['batch_size']
        dummy_batch = []
        dummy_input = np.expand_dims(np.random.rand(dims[0], dims[1], n_channels), axis=0)
        dummy_target = np.expand_dims(np.random.rand(1), axis=0)
        for i in range(batch_size):
            dummy_batch.append((dummy_input, dummy_target))
        collate_fn_output = collate_fn.__call__(dummy_batch)
        dummy_tensor = torch.rand(batch_size, n_channels, dims[0], dims[1])
        self.assertEqual(dummy_tensor.shape, collate_fn_output[0].shape)

    def test_model_train(self):
        train_params = {"max_epochs": 2, "lr_decay_factor": 0.1, "initial_lr": 0.025,
                        "loss": "cross_entropy",
                        "train_metrics_list": [Accuracy(), Top5()],
                        "valid_metrics_list": [Accuracy(), Top5()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}

        arch_params = {'num_classes': 1000}
        trainer = Trainer("test", model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(dataset_interface=self.test_external_dataset_interface,
                                          data_loader_num_workers=8)
        trainer.build_model("resnet50", arch_params)
        trainer.train(training_params=train_params)


if __name__ == '__main__':
    unittest.main()
