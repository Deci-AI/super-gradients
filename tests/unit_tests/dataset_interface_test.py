import unittest

from super_gradients.training.datasets import Cifar10DatasetInterface
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import PascalVOCUnifiedDetectionDataSetInterfaceV2
from super_gradients.training.transforms.transforms import DetectionPaddedRescale, DetectionTargetsFormatTransform, DetectionMosaic, DetectionRandomAffine,\
    DetectionHSV
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat
from super_gradients.training.utils.detection_utils import DetectionCollateFN
from super_gradients.training.utils import sg_model_utils
from super_gradients.training import utils as core_utils


class TestDatasetInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.ROOT_DIR = "/home/louis.dupont/data"
        self.TRAIN_BATCH_SIZE, self.VAL_BATCH_SIZE = 16, 32
        self.TRAIN_IMG_SIZE, self.VAL_IMG_SIZE = 640, 640
        self.TRAIN_INPUT_DIM = (self.TRAIN_IMG_SIZE, self.TRAIN_IMG_SIZE)
        self.VAL_INPUT_DIM = (self.VAL_IMG_SIZE, self.VAL_IMG_SIZE)

    def test_cifar(self):
        test_dataset_interface = Cifar10DatasetInterface()
        cifar_dataset_sample = test_dataset_interface.get_test_sample()
        self.assertListEqual([3, 32, 32], list(cifar_dataset_sample[0].shape))

    def setup_pascal_voc_interface_v2(self):
        """setup PascalVOCUnifiedDetectionDataSetInterfaceV2 and return dataloaders"""
        dataset_params = {
            "data_dir": self.ROOT_DIR + "/pascal_unified_coco_format/",
            "batch_size": self.TRAIN_BATCH_SIZE,
            "val_batch_size": self.VAL_BATCH_SIZE,
            "train_image_size": self.TRAIN_IMG_SIZE,
            "val_image_size": self.VAL_IMG_SIZE,
            "train_transforms": [
                DetectionMosaic(input_dim=self.TRAIN_INPUT_DIM, prob=1),
                DetectionRandomAffine(degrees=0.373, translate=0.245, scales=0.898, shear=0.602, target_size=self.TRAIN_INPUT_DIM),
                DetectionHSV(prob=1, hgain=0.0138, sgain=0.664, vgain=0.464),
                DetectionPaddedRescale(input_dim=self.TRAIN_INPUT_DIM, max_targets=100),
                DetectionTargetsFormatTransform(input_format=DetectionTargetsFormat.XYXY_LABEL,
                                                output_format=DetectionTargetsFormat.LABEL_CXCYWH)],
            "val_transforms": [
                DetectionPaddedRescale(input_dim=self.VAL_INPUT_DIM),
                DetectionTargetsFormatTransform(input_format=DetectionTargetsFormat.XYXY_LABEL,
                                                output_format=DetectionTargetsFormat.LABEL_CXCYWH)],
            "train_collate_fn": DetectionCollateFN(),
            "val_collate_fn": DetectionCollateFN(),
            "download": True,
            "cache_train_images": True,
            "cache_val_images": True,
            "class_inclusion_list": ["person"]
        }
        dataset_interface = PascalVOCUnifiedDetectionDataSetInterfaceV2(dataset_params=dataset_params)
        train_loader, valid_loader, _test_loader, _classes = dataset_interface.get_data_loaders()
        return train_loader, valid_loader

    def test_pascal_voc_v2(self):
        """Check that the dataset interface is correctly instantiated, and that the batch items are of expected size"""
        train_loader, valid_loader = self.setup_pascal_voc_interface_v2()

        for loader, batch_size, image_size in [(train_loader, self.TRAIN_BATCH_SIZE, self.TRAIN_IMG_SIZE),
                                               (valid_loader, self.VAL_BATCH_SIZE, self.VAL_IMG_SIZE)]:

            batch_items = next(iter(loader))
            batch_items = core_utils.tensor_container_to_device(batch_items, 'cuda', non_blocking=True)

            inputs, targets, additional_batch_items = sg_model_utils.unpack_batch_items(batch_items)
            self.assertListEqual([batch_size, 3, image_size, image_size], list(inputs.shape))


if __name__ == '__main__':
    unittest.main()
