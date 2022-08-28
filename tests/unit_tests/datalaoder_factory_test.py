import unittest

from torch.utils.data import DataLoader

from super_gradients.training.dataloaders.dataloader_factory import coco2017_train, coco2017_val, \
    coco2017_train_ssd_lite_mobilenet_v2, coco2017_val_ssd_lite_mobilenet_v2, pascal_voc_train, pascal_voc_val
from super_gradients.training.datasets import COCODetectionDataset
from super_gradients.training.datasets.detection_datasets.pascal_voc_detection import \
    PascalVOCUnifiedDetectionTrainDataset, PascalVOCDetectionDataset


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

    def test_pascal_voc_train_creation(self):
        dl_train = pascal_voc_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, PascalVOCUnifiedDetectionTrainDataset))

    def test_pascal_voc_val_creation(self):
        dl_val = pascal_voc_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, PascalVOCDetectionDataset))


if __name__ == '__main__':
    unittest.main()
