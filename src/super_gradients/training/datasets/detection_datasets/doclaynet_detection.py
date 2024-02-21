from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset

logger = get_logger(__name__)


@register_dataset(Datasets.DOCLAYNET_DETECTION_DATASET)
class DocLayNetDetectionDataset(COCOFormatDetectionDataset):
    """To use this Dataset you need to:

    - Download DocLayNetDataset dataset: https://github.com/DS4SD/DocLayNet

    - Unzip and organize it as below:
        DocLayNet
        ├── COCO
        │      ├─ test.json
        │      ├─ train.json
        │      └─ val.json
        └── PNG
            ├─ <image_name_1>.png
            ├─ <image_name_2>.png
            └─ ...

    - Install CoCo API: https://github.com/pdollar/coco/tree/master/PythonAPI

    - Instantiate the dataset:
        >> train_set = DocLayNetDetectionDataset(data_dir='.../DocLayNet', subdir='PNG', json_file='COCO/train.json', ...)
        >> valid_set = DocLayNetDetectionDataset(data_dir='.../DocLayNet', subdir='PNG', json_file='COCO/val.json', ...)
    """

    def __init__(
        self,
        json_file: str = "COCO/train.json",
        images_dir: str = "PNG",
        *args,
        **kwargs,
    ):
        """
        :param json_file:    Name of the coco json file, that resides in data_dir/annotations/json_file.
        :param images_dir:   Sub directory of data_dir containing the data.

        kwargs:
            with_crowd: Add the crowd groundtruths to __getitem__
            all_classes_list: all classes list, default is COCO_DETECTION_CLASSES_LIST.
        """
        kwargs.pop("subdir", None)
        kwargs.pop("root", None)
        if "json_annotation_file" in kwargs:
            json_file = kwargs["json_annotation_file"]
            kwargs.pop("json_annotation_file", None)

        super().__init__(json_annotation_file=json_file, images_dir=images_dir, *args, **kwargs)
