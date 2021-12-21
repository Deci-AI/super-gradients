from super_gradients.training.datasets import CoCoDetectionDatasetInterface
from super_gradients.training.utils.detection_utils import AnchorGenerator
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import PascalVOC2012DetectionDataSetInterface
dataset_params = {"batch_size": 1,
                  "test_batch_size": 1,
                  "image_size": 320,
                  }

coco_dataset_interface = PascalVOC2012DetectionDataSetInterface(dataset_params=dataset_params, cache_labels=True)
anchors = AnchorGenerator(coco_dataset_interface.trainset, 9)
