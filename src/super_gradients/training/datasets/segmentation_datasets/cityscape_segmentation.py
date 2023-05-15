from typing import List
import os
import cv2
import numpy as np
from PIL import Image, ImageColor
from torch.utils.data import ConcatDataset

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet


# label for background and labels to ignore during training and evaluation.
CITYSCAPES_IGNORE_LABEL = 19


@register_dataset(Datasets.CITYSCAPES_DATASET)
class CityscapesDataset(SegmentationDataSet):
    """
    CityscapesDataset - Segmentation Data Set Class for Cityscapes Segmentation Data Set,
    main resolution of dataset: (2048 x 1024).
    Not all the original labels are used for training and evaluation, according to cityscape paper:
    "Classes that are too rare are excluded from our benchmark, leaving 19 classes for evaluation".
    For more details about the dataset labels format see:
    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

    To use this Dataset you need to:

    - Download cityscape dataset (https://www.cityscapes-dataset.com/downloads/)

        root_dir (in recipe default to /data/cityscapes)
            ├─── gtFine
            │       ├── test
            │       │     ├── berlin
            │       │     │   ├── berlin_000000_000019_gtFine_color.png
            │       │     │   ├── berlin_000000_000019_gtFine_instanceIds.png
            │       │     │   └── ...
            │       │     ├── bielefeld
            │       │     │   └── ...
            │       │     └── ...
            │       ├─── train
            │       │     └── ...
            │       └─── val
            │             └── ...
            └─── leftImg8bit
                    ├── test
                    │     └── ...
                    ├─── train
                    │     └── ...
                    └─── val
                          └── ...

    - Download metadata folder (https://deci-pretrained-models.s3.amazonaws.com/cityscape_lists.zip)

        lists
            ├── labels.csv
            ├── test.lst
            ├── train.lst
            ├── trainval.lst
            └── val.lst

    - Move Metadata folder to the Cityscape folder

        root_dir (in recipe default to /data/cityscapes)
            ├─── gtFine
            │      └── ...
            ├─── leftImg8bit
            │      └── ...
            └─── lists
                   └── ...

    Example:
        >> CityscapesDataset(root_dir='.../root_dir', list_file='lists/train.lst', labels_csv_path='lists/labels.csv', ...)
    """

    def __init__(self, root_dir: str, list_file: str, labels_csv_path: str, **kwargs):
        """
        :param root_dir:        Absolute path to root directory of the dataset.
        :param list_file:       List file that contains names of images to load, line format: <image_path> <label_path>. The path is relative to root.
        :param labels_csv_path: Path to csv file, with labels metadata and mapping. The path is relative to root.
        :param kwargs:          Any hyper params required for the dataset, i.e img_size, crop_size, cache_images
        """

        self.root_dir = root_dir
        super().__init__(root_dir, list_file=list_file, **kwargs)
        # labels dataframe for labels metadata.
        self.labels_data = np.recfromcsv(os.path.join(self.root_dir, labels_csv_path), dtype="<i8,U20,<i8,<i8,U12,<i8,?,?,U7", comments="&")
        # map vector to map ground-truth labels to train labels
        self.labels_map = self.labels_data.field("trainid")
        # class names
        self.classes = self.labels_data.field("name")[np.logical_not(self.labels_data.field("ignoreineval"))].tolist()
        # color palette for visualization
        self.train_id_color_palette = self._create_color_palette()

    def _generate_samples_and_targets(self):
        """
        override _generate_samples_and_targets function, to parse list file.
        line format of list file: <image_path> <label_path>
        """
        with open(os.path.join(self.root_dir, self.list_file_path)) as f:
            img_list = [line.strip().split() for line in f]
        for image_path, label_path in img_list:
            self.samples_targets_tuples_list.append((os.path.join(self.root, image_path), os.path.join(self.root, label_path)))
        super(CityscapesDataset, self)._generate_samples_and_targets()

    def target_loader(self, label_path: str) -> Image:
        """
        Override target_loader function, load the labels mask image.
            :param label_path:  Path to the label image.
            :return:                     The mask image created from the array, with converted class labels.
        """
        # assert that is a png file, other file types might alter the class labels value.
        assert os.path.splitext(label_path)[-1].lower() == ".png"

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # map ground-truth ids to train ids
        label = self.labels_map[label].astype(np.uint8)
        return Image.fromarray(label, "L")

    def _create_color_palette(self):
        """
        Create color pallete for visualizing the segmentation masks
        :return: list of rgb color values
        """
        palette = []
        hex_colors = self.labels_data.field("color")[np.logical_not(self.labels_data.field("ignoreineval"))].tolist()

        for hex_color in hex_colors:
            rgb_color = ImageColor.getcolor(hex_color, "RGB")
            palette += [x for x in rgb_color]

        return palette

    def get_train_ids_color_palette(self):
        return self.train_id_color_palette

    @staticmethod
    def target_transform(target):
        """
        target_transform - Transforms the sample image
        This function overrides the original function from SegmentationDataSet and changes target pixels with value
        255 to value = CITYSCAPES_IGNORE_LABEL. This was done since current IoU metric from torchmetrics does not
        support such a high ignore label value (crashed on OOM)

            :param target: The target mask to transform
            :return:       The transformed target mask
        """
        out = SegmentationDataSet.target_transform(target)
        out[out == 255] = CITYSCAPES_IGNORE_LABEL
        return out


@register_dataset(Datasets.CITYSCAPES_CONCAT_DATASET)
class CityscapesConcatDataset(ConcatDataset):
    """
    Support building a Cityscapes dataset which includes multiple group of samples from several list files.
    i.e to initiate a trainval dataset:
    >>> trainval_set = CityscapesConcatDataset(
    >>>    root_dir='/data', list_files=['lists/train.lst', 'lists/val.lst'], labels_csv_path='lists/labels.csv', ...
    >>> )

    i.e to initiate a combination of the train-set with AutoLabelling-set:
    >>> train_al_set = CityscapesConcatDataset(
    >>>    root_dir='/data', list_files=['lists/train.lst', 'lists/auto_labelling.lst'], labels_csv_path='lists/labels.csv', ...
    >>> )
    """

    def __init__(self, root_dir: str, list_files: List[str], labels_csv_path: str, **kwargs):
        """
        :param root_dir:        Absolute path to root directory of the dataset.
        :param list_files:      List of list files that contains names of images to load,
                                line format: <image_path> <label_path>. The path is relative to root.
        :param labels_csv_path: Path to csv file, with labels metadata and mapping. The path is relative to root.
        :param kwargs:          Any hyper params required for the dataset, i.e img_size, crop_size, cache_images
        """
        super().__init__(
            datasets=[
                CityscapesDataset(
                    root_dir=root_dir,
                    list_file=list_file,
                    labels_csv_path=labels_csv_path,
                    **kwargs,
                )
                for list_file in list_files
            ]
        )
