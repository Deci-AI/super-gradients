import unittest

from torch.utils.data import DataLoader, TensorDataset

from super_gradients.training.dataloaders.dataloader_factory import coco2017_train, coco2017_val, \
    coco2017_train_ssd_lite_mobilenet_v2, coco2017_val_ssd_lite_mobilenet_v2, classification_test_dataloader, detection_test_dataloader,\
    segmentation_test_dataloader
from super_gradients.training.datasets import COCODetectionDataset


class DataLoaderFactoryTest(unittest.TestCase):
    def test_coco2017_train_creation(self):
        dl_train = coco2017_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, COCODetectionDataset))

    def test_coco2017_val_creation(self):
        dl_val = coco2017_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, COCODetectionDataset))

    def test_coco2017_train_ssdlite_mobilenet_creation(self):
        dl_train = coco2017_train_ssd_lite_mobilenet_v2()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, COCODetectionDataset))

    def test_coco2017_val_ssdlite_mobilenet_creation(self):
        dl_train = coco2017_val_ssd_lite_mobilenet_v2()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, COCODetectionDataset))

    def test_classification_test_creation(self):
        dl = classification_test_dataloader()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, TensorDataset))

    def test_detection_test_creation(self):
        dl = detection_test_dataloader()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, TensorDataset))

    def test_segmentation_test_creation(self):
        dl = segmentation_test_dataloader()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, TensorDataset))


if __name__ == '__main__':
    unittest.main()
