import unittest

from torch.utils.data import DataLoader, TensorDataset

from super_gradients.training.dataloaders.dataloader_factory import (
    classification_test_dataloader,
    detection_test_dataloader,
    segmentation_test_dataloader,
    coco2017_train,
    coco2017_val,
    coco2017_train_ssd_lite_mobilenet_v2,
    coco2017_val_ssd_lite_mobilenet_v2,
    imagenet_train,
    imagenet_val,
    imagenet_efficientnet_train,
    imagenet_efficientnet_val,
    imagenet_mobilenetv2_train,
    imagenet_mobilenetv2_val,
    imagenet_mobilenetv3_train,
    imagenet_mobilenetv3_val,
    imagenet_regnetY_train,
    imagenet_regnetY_val,
    imagenet_resnet50_train,
    imagenet_resnet50_val,
    imagenet_resnet50_kd_train,
    imagenet_resnet50_kd_val,
    imagenet_vit_base_train,
    imagenet_vit_base_val,
    tiny_imagenet_train,
    tiny_imagenet_val,
    cityscapes_train, cityscapes_val,
    cityscapes_stdc_seg50_train, cityscapes_stdc_seg50_val, cityscapes_stdc_seg75_val, cityscapes_ddrnet_train,
    cityscapes_regseg48_val, cityscapes_regseg48_train, cityscapes_ddrnet_val, cityscapes_stdc_seg75_train
)
from super_gradients.training.datasets import COCODetectionDataset, ImageNetDataset
from super_gradients.training.datasets.segmentation_datasets import CityscapesDataset


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

    def test_imagenet_train_creation(self):
        dl = imagenet_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_val_creation(self):
        dl = imagenet_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_efficientnet_train_creation(self):
        dl = imagenet_efficientnet_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_efficientnet_val_creation(self):
        dl = imagenet_efficientnet_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_mobilenetv2_train_creation(self):
        dl = imagenet_mobilenetv2_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_mobilenetv2_val_creation(self):
        dl = imagenet_mobilenetv2_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_mobilenetv3_train_creation(self):
        dl = imagenet_mobilenetv3_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_mobilenetv3_val_creation(self):
        dl = imagenet_mobilenetv3_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_regnetY_train_creation(self):
        dl = imagenet_regnetY_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_regnetY_val_creation(self):
        dl = imagenet_regnetY_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_resnet50_train_creation(self):
        dl = imagenet_resnet50_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_resnet50_val_creation(self):
        dl = imagenet_resnet50_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_resnet50_kd_train_creation(self):
        # Here we need to overwrite the sampler because the RepeatAugSampler used in KD is only supported for DDP
        dl = imagenet_resnet50_kd_train(dataloader_params={"sampler": {"InfiniteSampler": {}}})
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_resnet50_kd_val_creation(self):
        dl = imagenet_resnet50_kd_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_vit_base_train_creation(self):
        dl = imagenet_vit_base_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_vit_base_val_creation(self):
        dl = imagenet_vit_base_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_tiny_imagenet_train_train_creation(self):
        dl = tiny_imagenet_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_tiny_imagenet_train_val_creation(self):
        dl = tiny_imagenet_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_classification_test_dataloader_creation(self):
        dl = classification_test_dataloader()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, TensorDataset))

    def test_detection_test_dataloader_creation(self):
        dl = detection_test_dataloader()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, TensorDataset))

    def test_segmentation_test_dataloader_creation(self):
        dl = segmentation_test_dataloader()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, TensorDataset))

    def test_cityscapes_train_creation(self):
        dl_train = cityscapes_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, CityscapesDataset))
        it = iter(dl_train)
        for _ in range(10):
            next(it)

    def test_cityscapes_val_creation(self):
        dl_val = cityscapes_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, CityscapesDataset))
        it = iter(dl_val)
        for _ in range(10):
            next(it)

    def test_cityscapes_stdc_seg50_train_creation(self):
        dl_train = cityscapes_stdc_seg50_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, CityscapesDataset))
        it = iter(dl_train)
        for _ in range(10):
            next(it)

    def test_cityscapes_stdc_seg50_val_creation(self):
        dl_val = cityscapes_stdc_seg50_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, CityscapesDataset))
        it = iter(dl_val)
        for _ in range(10):
            next(it)

    def test_cityscapes_stdc_seg75_train_creation(self):
        dl_train = cityscapes_stdc_seg75_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, CityscapesDataset))
        it = iter(dl_train)
        for _ in range(10):
            next(it)

    def test_cityscapes_stdc_seg75_val_creation(self):
        dl_val = cityscapes_stdc_seg75_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, CityscapesDataset))
        it = iter(dl_val)
        for _ in range(10):
            next(it)

    def test_cityscapes_regseg48_train_creation(self):
        dl_train = cityscapes_regseg48_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, CityscapesDataset))
        it = iter(dl_train)
        for _ in range(10):
            next(it)

    def test_cityscapes_regseg48_val_creation(self):
        dl_val = cityscapes_regseg48_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, CityscapesDataset))
        it = iter(dl_val)
        for _ in range(10):
            next(it)

    def test_cityscapes_ddrnet_train_creation(self):
        dl_train = cityscapes_ddrnet_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, CityscapesDataset))
        it = iter(dl_train)
        for _ in range(10):
            next(it)

    def test_cityscapes_ddrnet_val_creation(self):
        dl_val = cityscapes_ddrnet_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, CityscapesDataset))
        it = iter(dl_val)
        for _ in range(10):
            next(it)


if __name__ == '__main__':
    unittest.main()
