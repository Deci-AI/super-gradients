import csv
import os

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet


@register_dataset(Datasets.SUPERVISELY_PERSONS_DATASET)
class SuperviselyPersonsDataset(SegmentationDataSet):
    """
    SuperviselyPersonsDataset - Segmentation Data Set Class for Supervisely Persons Segmentation Data Set,
    main resolution of dataset: (600 x 800).
    This dataset is a subset of the original dataset (see below) and contains filtered samples
    For more details about the ORIGINAL dataset see:
        https://app.supervise.ly/ecosystem/projects/persons
    For more details about the FILTERED dataset see:
        https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.3/contrib/PP-HumanSeg

    To use this Dataset you need to:

        - Download supervisely dataset:
            https://deci-pretrained-models.s3.amazonaws.com/supervisely-persons.zip)

        - Unzip:
            supervisely-persons
             ├──images
             │    ├──image-name.png
             │    └──...
             ├──images_600x800
             │    ├──image-name.png
             │    └──...
             ├──masks
             └──masks_600x800

        - Instantiate the dataset:
            >> train_set = SuperviselyPersonsDataset(root_dir='.../supervisely-persons', list_file='train.csv', ...)
            >> valid_set = SuperviselyPersonsDataset(root_dir='.../supervisely-persons', list_file='val.csv', ...)
    """

    CLASS_LABELS = {0: "background", 1: "person"}

    def __init__(self, root_dir: str, list_file: str, **kwargs):
        """
        :param root_dir:    root directory to dataset.
        :param list_file:   list file that contains names of images to load, line format: <image_path>,<mask_path>
        :param kwargs:      Any hyper params required for the dataset, i.e img_size, crop_size, etc...
        """

        super().__init__(root=root_dir, list_file=list_file, **kwargs)
        self.classes = ["person"]

    def _generate_samples_and_targets(self):
        with open(os.path.join(self.root, self.list_file_path), "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                sample_path = os.path.join(self.root, row[0])
                target_path = os.path.join(self.root, row[1])
                if self._validate_file(sample_path) and self._validate_file(target_path) and os.path.exists(sample_path) and os.path.exists(target_path):
                    self.samples_targets_tuples_list.append((sample_path, target_path))
                else:
                    raise AssertionError(
                        f"Sample and/or target file(s) not found or in illegal format " f"(sample path: {sample_path}, target path: {target_path})"
                    )
        super(SuperviselyPersonsDataset, self)._generate_samples_and_targets()
