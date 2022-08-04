import unittest

from super_gradients.training.datasets import Cifar10DatasetInterface
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import PascalVOCUnifiedDetectionDatasetInterface,\
    CoCoDetectionDatasetInterface
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

    def test_cifar(self):
        test_dataset_interface = Cifar10DatasetInterface()
        cifar_dataset_sample = test_dataset_interface.get_test_sample()
        self.assertListEqual([3, 32, 32], list(cifar_dataset_sample[0].shape))

    def setup_pascal_voc_interface(self):
        """setup PascalVOCUnifiedDetectionDatasetInterface and return dataloaders"""
        dataset_params = {
            "data_dir": self.root_dir + "pascal_unified_coco_format/",
            "cache_dir": self.root_dir + "pascal_unified_coco_format/",
            "batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "train_image_size": self.train_image_size,
            "val_image_size": self.val_image_size,
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

    def setup_coco_detection_interface(self):
        """setup CoCoDetectionDatasetInterface and return dataloaders"""
        dataset_params = {
            "data_dir": "/data/coco",
            "train_subdir": "images/train2017", # sub directory path of data_dir containing the train data.
            "val_subdir": "images/val2017", # sub directory path of data_dir containing the validation data.
            "train_json_file": "instances_train2017.json", # path to coco train json file, data_dir/annotations/train_json_file.
            "val_json_file": "instances_val2017.json", # path to coco validation json file, data_dir/annotations/val_json_file.

            "batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "train_image_size": self.train_image_size,
            "val_image_size": self.val_image_size,

            "mixup_prob": 1.0,  # probability to apply per-sample mixup
            "degrees": 10.,  # rotation degrees, randomly sampled from [-degrees, degrees]
            "shear": 2.0,  # shear degrees, randomly sampled from [-degrees, degrees]
            "flip_prob": 0.5,  # probability to apply horizontal flip
            "hsv_prob": 1.0,  # probability to apply HSV transform
            "hgain": 5,  # HSV transform hue gain (randomly sampled from [-hgain, hgain])
            "sgain": 30,  # HSV transform saturation gain (randomly sampled from [-sgain, sgain])
            "vgain": 30,  # HSV transform value gain (randomly sampled from [-vgain, vgain])
            "mosaic_scale": [0.1, 2],  # random rescale range (keeps size by padding/cropping) after mosaic transform.
            "mixup_scale": [0.5, 1.5],  # random rescale range for the additional sample in mixup
            "mosaic_prob": 1.,  # probability to apply mosaic
            "translate": 0.1,  # image translation fraction
            "filter_box_candidates": False,  # whether to filter out transformed bboxes by edge size, area ratio, and aspect ratio.
            "wh_thr": 2,  # edge size threshold when filter_box_candidates = True (pixels)
            "ar_thr": 20,  # aspect ratio threshold when filter_box_candidates = True
            "area_thr": 0.1,  # threshold for area ratio between original image and the transformed one, when when filter_box_candidates = True

            "download": True,
            "train_collate_fn": DetectionCollateFN(),
            "val_collate_fn": DetectionCollateFN(),
            "cache_train_images": False,
            "cache_val_images": False,
            "cache_dir_path": "/home/louis.dupont/data/cache",
            # "with_crowd": True
        }

        dataset_interface = CoCoDetectionDatasetInterface(dataset_params=dataset_params)
        train_loader, valid_loader, _test_loader, _classes = dataset_interface.get_data_loaders()
        return train_loader, valid_loader

    def test_coco_detection(self):
        """Check that the dataset interface is correctly instantiated, and that the batch items are of expected size"""
        train_loader, valid_loader = self.setup_coco_detection_interface()

        for loader, batch_size, image_size in [(train_loader, self.train_batch_size, self.train_image_size),
                                               (valid_loader, self.val_batch_size, self.val_image_size)]:

            batch_items = next(iter(loader))
            batch_items = core_utils.tensor_container_to_device(batch_items, 'cuda', non_blocking=True)

            inputs, targets, additional_batch_items = sg_model_utils.unpack_batch_items(batch_items)
            self.assertListEqual([batch_size, 3, image_size, image_size], list(inputs.shape))

    def test_pascal_voc(self):
        """Check that the dataset interface is correctly instantiated, and that the batch items are of expected size"""
        train_loader, valid_loader = self.setup_pascal_voc_interface()

        for loader, batch_size, image_size in [(train_loader, self.train_batch_size, self.train_image_size),
                                               (valid_loader, self.val_batch_size, self.val_image_size)]:

            batch_items = next(iter(loader))
            batch_items = core_utils.tensor_container_to_device(batch_items, 'cuda', non_blocking=True)

            inputs, targets, additional_batch_items = sg_model_utils.unpack_batch_items(batch_items)
            self.assertListEqual([batch_size, 3, image_size, image_size], list(inputs.shape))


if __name__ == '__main__':
    unittest.main()
