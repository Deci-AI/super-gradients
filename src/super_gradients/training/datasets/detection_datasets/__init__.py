from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataSet
from super_gradients.training.datasets.detection_datasets.coco_detection import COCODetectionDataSet
from super_gradients.training.datasets.detection_datasets.coco_detection_yolox import COCODetectionDatasetV2
from super_gradients.training.datasets.detection_datasets.pascal_voc_detection import PascalVOCDetectionDataSet
from super_gradients.training.datasets.detection_datasets.pascal_voc_detection_v2 import PascalVOCDetectionDataSetV2
from super_gradients.training.datasets.detection_datasets.detection_dataset_v2 import DetectionDataSetV2


__all__ = ['DetectionDataSet', 'COCODetectionDataSet', 'DetectionDataSetV2', 'COCODetectionDatasetV2',
           'PascalVOCDetectionDataSet', 'PascalVOCDetectionDataSetV2']
DetectionDataSetV2 = 1