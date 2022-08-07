import unittest

from super_gradients.training.datasets.dataset_interfaces.dataset_interface import PascalVOCUnifiedDetectionDatasetInterface
from super_gradients.training.transforms.transforms import DetectionPaddedRescale, DetectionTargetsFormatTransform, DetectionMosaic, DetectionRandomAffine,\
    DetectionHSV
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat
from super_gradients.training.utils.detection_utils import DetectionCollateFN
from super_gradients.training.utils import sg_model_utils
from super_gradients.training import utils as core_utils


class TestDatasetInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.root_dir = "/home/louis.dupont/data/"
        self.train_batch_size, self.val_batch_size = 16, 32
        self.train_image_size, self.val_image_size = 640, 640
        self.train_input_dim = (self.train_image_size, self.train_image_size)
        self.val_input_dim = (self.val_image_size, self.val_image_size)
        self.train_max_num_samples = 100
        self.val_max_num_samples = 90

    def setup_pascal_voc_interface(self):
        """setup PascalVOCUnifiedDetectionDataSetInterfaceV2 and return dataloaders"""
        dataset_params = {
            "data_dir": self.root_dir + "pascal_unified_coco_format/",
            "cache_dir": self.root_dir + "pascal_unified_coco_format/",
            "batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "train_image_size": self.train_image_size,
            "val_image_size": self.val_image_size,
            "train_max_num_samples": self.train_max_num_samples,
            "val_max_num_samples": self.val_max_num_samples,
            "train_transforms": [
                DetectionMosaic(input_dim=self.train_input_dim, prob=1),
                DetectionRandomAffine(degrees=0.373, translate=0.245, scales=0.898, shear=0.602, target_size=self.train_input_dim),
                DetectionHSV(prob=1, hgain=0.0138, sgain=0.664, vgain=0.464),
                DetectionPaddedRescale(input_dim=self.train_input_dim, max_targets=100),
                DetectionTargetsFormatTransform(input_format=DetectionTargetsFormat.XYXY_LABEL,
                                                output_format=DetectionTargetsFormat.LABEL_CXCYWH)],
            "val_transforms": [
                DetectionPaddedRescale(input_dim=self.val_input_dim),
                DetectionTargetsFormatTransform(input_format=DetectionTargetsFormat.XYXY_LABEL,
                                                output_format=DetectionTargetsFormat.LABEL_CXCYWH)],
            "train_collate_fn": DetectionCollateFN(),
            "val_collate_fn": DetectionCollateFN(),
            "download": False,
            "cache_train_images": False,
            "cache_val_images": False,
            "class_inclusion_list": ["person"]
        }
        dataset_interface = PascalVOCUnifiedDetectionDatasetInterface(dataset_params=dataset_params)
        train_loader, valid_loader, _test_loader, _classes = dataset_interface.get_data_loaders()
        return train_loader, valid_loader

    def test_pascal_voc(self):
        """Check that the dataset interface is correctly instantiated, and that the batch items are of expected size"""
        train_loader, valid_loader = self.setup_pascal_voc_interface()

        for loader, batch_size, image_size, max_num_samples in [(train_loader, self.train_batch_size, self.train_image_size, self.train_max_num_samples),
                                                                (valid_loader, self.val_batch_size, self.val_image_size, self.val_max_num_samples)]:
            # The dataset is at most of length max_num_samples, but can be smaller if not enough samples
            self.assertGreaterEqual(max_num_samples, len(loader.dataset))

            batch_items = next(iter(loader))
            batch_items = core_utils.tensor_container_to_device(batch_items, 'cuda', non_blocking=True)

            inputs, targets, additional_batch_items = sg_model_utils.unpack_batch_items(batch_items)
            self.assertListEqual([batch_size, 3, image_size, image_size], list(inputs.shape))


if __name__ == '__main__':
    unittest.main()
