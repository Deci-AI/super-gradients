from super_gradients.training.datasets import CoCoDetectionDatasetInterface
from super_gradients.training.utils.detection_utils import AnchorGenerator

dataset_params = {"batch_size": 1,
                  "test_batch_size": 1,
                  "dataset_dir": "/data/coco/",
                  "image_size": 640,
                  "train_sample_loading_method": 'rectangular'
                  }

coco_dataset_interface = CoCoDetectionDatasetInterface(dataset_params=dataset_params, cache_labels=True)
ag = AnchorGenerator()
anchors = ag(dataset = coco_dataset_interface.trainset)
print(anchors)