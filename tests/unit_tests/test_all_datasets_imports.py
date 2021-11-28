from unittest import TestCase

from super_gradients.training.datasets.all_datasets import SgLibraryDatasets, DatasetInterface


class TestSgLibraryDatasets(TestCase):
    def setUp(self) -> None:
        self.sg_library_datasets = SgLibraryDatasets

    def test_get_all_datasets(self):
        all_datasets = self.sg_library_datasets.get_all_available_datasets()
        self.assertIsInstance(all_datasets, dict)

    def test_get_dateset(self):
        cifar_100 = self.sg_library_datasets.get_dataset('classification', 'cifar_100')
        self.assertTrue(issubclass(cifar_100, DatasetInterface))

    def test_get_dateset_with_invalid_dataset_name_raises_exception(self):
        with self.assertRaises(ValueError):
            self.sg_library_datasets.get_dataset('classification',
                                                   'cifar_1000000')

    def test_get_dateset_with_invalid_dl_task_raises_exception(self):
        with self.assertRaises(ValueError):
            self.sg_library_datasets.get_dataset('classification_of_something_that_deci_does_not_support',
                                                   'cifar_100')
