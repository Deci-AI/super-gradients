import unittest

from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.dataloaders.dataloaders import (
    classification_test_dataloader,
    detection_test_dataloader,
    segmentation_test_dataloader,
    cifar10_val,
    cifar10_train,
    cifar100_val,
    cifar100_train,
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
    pascal_aug_segmentation_train,
    pascal_aug_segmentation_val,
    pascal_voc_segmentation_train,
    pascal_voc_segmentation_val,
    supervisely_persons_train,
    supervisely_persons_val,
    pascal_voc_detection_train,
    pascal_voc_detection_val,
    get,
    mapillary_train,
    mapillary_val,
)
from super_gradients.training.datasets import (
    COCODetectionDataset,
    ImageNetDataset,
    PascalVOC2012SegmentationDataSet,
    SuperviselyPersonsDataset,
    PascalVOCDetectionDataset,
    Cifar10,
    Cifar100,
    PascalVOCAndAUGUnifiedDataset,
)
import torch
import numpy as np

from super_gradients.training.datasets.detection_datasets.pascal_voc_detection import PascalVOCUnifiedDetectionTrainDataset
from super_gradients.training.datasets.segmentation_datasets import MapillaryDataset
from super_gradients import init_trainer


@register_dataset("FixedLenDataset")
class FixedLenDataset(TensorDataset):
    def __init__(self):
        images = torch.Tensor(np.zeros((10, 3, 32, 32)))
        ground_truth = torch.LongTensor(np.zeros((10)))
        super(FixedLenDataset, self).__init__(images, ground_truth)


class DataLoaderFactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        init_trainer()

    def test_coco2017_train_creation(self):
        dl_train = coco2017_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, COCODetectionDataset))
        self.assertTrue(dl_train.batch_sampler.sampler._shuffle)

    def test_coco2017_val_creation(self):
        dl_val = coco2017_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, COCODetectionDataset))

    def test_coco2017_train_ssdlite_mobilenet_creation(self):
        dl_train = coco2017_train_ssd_lite_mobilenet_v2()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, COCODetectionDataset))
        self.assertTrue(dl_train.batch_sampler.sampler._shuffle)

    def test_coco2017_val_ssdlite_mobilenet_creation(self):
        dl_train = coco2017_val_ssd_lite_mobilenet_v2()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, COCODetectionDataset))

    def test_imagenet_train_creation(self):
        dl = imagenet_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_imagenet_val_creation(self):
        dl = imagenet_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_efficientnet_train_creation(self):
        dl = imagenet_efficientnet_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_imagenet_efficientnet_val_creation(self):
        dl = imagenet_efficientnet_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_mobilenetv2_train_creation(self):
        dl = imagenet_mobilenetv2_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_imagenet_mobilenetv2_val_creation(self):
        dl = imagenet_mobilenetv2_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_mobilenetv3_train_creation(self):
        dl = imagenet_mobilenetv3_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_imagenet_mobilenetv3_val_creation(self):
        dl = imagenet_mobilenetv3_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_regnetY_train_creation(self):
        dl = imagenet_regnetY_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_imagenet_regnetY_val_creation(self):
        dl = imagenet_regnetY_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_resnet50_train_creation(self):
        dl = imagenet_resnet50_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_imagenet_resnet50_val_creation(self):
        dl = imagenet_resnet50_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, ImageNetDataset))

    def test_imagenet_resnet50_kd_train_creation(self):
        # Here we need to overwrite the sampler because the RepeatAugSampler used in KD is only supported for DDP
        dl = imagenet_resnet50_kd_train(dataloader_params={"sampler": {"RandomSampler": {}}})
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
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

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

    def test_cifar10_train_creation(self):
        dl_train = cifar10_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, Cifar10))
        self.assertTrue(isinstance(dl_train.sampler, RandomSampler))

    def test_cifar10_val_creation(self):
        dl_val = cifar10_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, Cifar10))

    def test_cifar100_train_creation(self):
        dl_train = cifar100_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, Cifar100))
        self.assertTrue(isinstance(dl_train.sampler, RandomSampler))

    def test_cifar100_val_creation(self):
        dl_val = cifar100_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, Cifar100))

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

    def test_pascal_aug_segmentation_train_creation(self):
        dl = pascal_aug_segmentation_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, PascalVOCAndAUGUnifiedDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_pascal_aug_segmentation_val_creation(self):
        dl = pascal_aug_segmentation_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, PascalVOC2012SegmentationDataSet))

    def test_pascal_voc_segmentation_train_creation(self):
        dl = pascal_voc_segmentation_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, PascalVOC2012SegmentationDataSet))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_pascal_voc_segmentation_val_creation(self):
        dl = pascal_voc_segmentation_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, PascalVOC2012SegmentationDataSet))

    def test_supervisely_persons_train_dataloader_creation(self):
        dl = supervisely_persons_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, SuperviselyPersonsDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_supervisely_persons_val_dataloader_creation(self):
        dl = supervisely_persons_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, SuperviselyPersonsDataset))

    def test_pascal_voc_train_creation(self):
        dl = pascal_voc_detection_train()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, PascalVOCUnifiedDetectionTrainDataset))
        self.assertTrue(dl.batch_sampler.sampler._shuffle)

    def test_pascal_voc_val_creation(self):
        dl = pascal_voc_detection_val()
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, PascalVOCDetectionDataset))

    def test_mapillary_train_creation(self):
        dl_train = mapillary_train()
        self.assertTrue(isinstance(dl_train, DataLoader))
        self.assertTrue(isinstance(dl_train.dataset, MapillaryDataset))

    def test_mapillary_val_creation(self):
        dl_val = mapillary_val()
        self.assertTrue(isinstance(dl_val, DataLoader))
        self.assertTrue(isinstance(dl_val.dataset, MapillaryDataset))

    def test_get_with_external_dataset_creation(self):
        dataset = Cifar10(root="./data/cifar10", train=False, download=True)
        dl = get(dataset=dataset, dataloader_params={"batch_size": 256, "num_workers": 8, "drop_last": False, "pin_memory": True, "shuffle": True})
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))

    def test_get_with_registered_dataset(self):
        dl = get(dataloader_params={"dataset": "FixedLenDataset", "batch_size": 256, "num_workers": 8, "drop_last": False, "pin_memory": True, "shuffle": True})
        self.assertTrue(isinstance(dl.dataset, FixedLenDataset))
        self.assertTrue(isinstance(dl.sampler, RandomSampler))


if __name__ == "__main__":
    unittest.main()
