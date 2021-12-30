from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataSet
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST


class COCODetectionDataSet(DetectionDataSet):
    """
    COCODetectionDataSet - Detection Data Set Class COCO Data Set
    """

    def __init__(self, *args, **kwargs):
        kwargs['all_classes_list'] = COCO_DETECTION_CLASSES_LIST
        super().__init__(*args, **kwargs)
