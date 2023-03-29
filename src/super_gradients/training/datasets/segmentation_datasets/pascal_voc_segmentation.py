import os

import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import ConcatDataset

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

PASCAL_VOC_2012_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted-plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


@register_dataset(Datasets.PASCAL_VOC_2012_SEGMENTATION_DATASET)
class PascalVOC2012SegmentationDataSet(SegmentationDataSet):
    """
    Segmentation Data Set Class for Pascal VOC 2012 Data Set.

    To use this Dataset you need to:

        - Download pascal VOC 2012 dataset:
            http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

        - Unzip and organize it as below:
            pascal_voc_2012
                └──VOCdevkit
                      └──VOC2012
                         ├──JPEGImages
                         ├──SegmentationClass
                         ├──ImageSets
                         │    ├──Segmentation
                         │    │   └── train.txt
                         │    ├──Main
                         │    ├──Action
                         │    └──Layout
                         ├──Annotations
                         └──SegmentationObject

        - Instantiate the dataset:
            >> train_set = PascalVOC2012SegmentationDataSet(
                    root='.../pascal_voc_2012',
                    list_file='VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
                    samples_sub_directory='VOCdevkit/VOC2012/JPEGImages',
                    targets_sub_directory='VOCdevkit/VOC2012/SegmentationClass',
                    ...
                )
            >> valid_set = PascalVOC2012SegmentationDataSet(
                    root='.../pascal_voc_2012',
                    list_file='VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
                    samples_sub_directory='VOCdevkit/VOC2012/JPEGImages',
                    targets_sub_directory='VOCdevkit/VOC2012/SegmentationClass',
                    ...
                )
    """

    IGNORE_LABEL = 21
    _ORIGINAL_IGNORE_LABEL = 255

    def __init__(self, sample_suffix=None, target_suffix=None, *args, **kwargs):
        self.sample_suffix = ".jpg" if sample_suffix is None else sample_suffix
        self.target_suffix = ".png" if target_suffix is None else target_suffix
        super().__init__(*args, **kwargs)

        self.classes = PASCAL_VOC_2012_CLASSES

    @staticmethod
    def target_transform(target):
        """
        target_transform - Transforms the label mask
        This function overrides the original function from SegmentationDataSet and changes target pixels with value
        255 to value = IGNORE_LABEL. This was done since current IoU metric from torchmetrics does not
        support such a high ignore label value (crashed on OOM)

            :param target: The target mask to transform
            :return:       The transformed target mask
        """
        out = SegmentationDataSet.target_transform(target)
        out[out == PascalVOC2012SegmentationDataSet._ORIGINAL_IGNORE_LABEL] = PascalVOC2012SegmentationDataSet.IGNORE_LABEL
        return out

    def decode_segmentation_mask(self, label_mask: np.ndarray):
        """
        decode_segmentation_mask - Decodes the colors for the Segmentation Mask
            :param: label_mask:  an (M,N) array of integer values denoting
                                the class label at each spatial location.
        :return:
        """
        label_colours = self._get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()

        num_classes_to_plot = len(self.classes)
        for ll in range(0, num_classes_to_plot):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb

    def _generate_samples_and_targets(self):
        """
        _generate_samples_and_targets
        """
        # GENERATE SAMPLES AND TARGETS HERE SPECIFICALLY FOR PASCAL VOC 2012
        with open(self.root + os.path.sep + self.list_file_path, "r", encoding="utf-8") as lines:
            for line in lines:
                image_path = os.path.join(self.root, self.samples_sub_directory, line.rstrip("\n") + self.sample_suffix)
                mask_path = os.path.join(self.root, self.targets_sub_directory, line.rstrip("\n") + self.target_suffix)

                if os.path.exists(mask_path) and os.path.exists(image_path):
                    self.samples_targets_tuples_list.append((image_path, mask_path))

        # GENERATE SAMPLES AND TARGETS OF THE SEGMENTATION DATA SET CLASS
        super()._generate_samples_and_targets()

    def _get_pascal_labels(self) -> np.ndarray:
        """Load the mapping that associates pascal classes with label colors
        :return: np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )


@register_dataset(Datasets.PASCAL_AUG_2012_SEGMENTATION_DATASET)
class PascalAUG2012SegmentationDataSet(PascalVOC2012SegmentationDataSet):
    """
    Segmentation Data Set Class for Pascal AUG 2012 Data Set

        - Download pascal AUG 2012 dataset:
            https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

        - Unzip and organize it as below:
            pascal_voc_2012
                └──VOCaug
                    ├── aug.txt
                    └── dataset
                          ├──inst
                          ├──img
                          └──cls

        - Instantiate the dataset:
            >> train_set = PascalAUG2012SegmentationDataSet(
                    root='.../pascal_voc_2012',
                    list_file='VOCaug/dataset/aug.txt',
                    samples_sub_directory='VOCaug/dataset/img',
                    targets_sub_directory='VOCaug/dataset/cls',
                    ...
                )

    NOTE: this dataset is only available for training. To test, please use PascalVOC2012SegmentationDataSet.
    """

    def __init__(self, *args, **kwargs):
        self.sample_suffix = ".jpg"
        self.target_suffix = ".mat"
        super().__init__(sample_suffix=self.sample_suffix, target_suffix=self.target_suffix, *args, **kwargs)

    @staticmethod
    def target_loader(target_path: str) -> Image:
        """
        target_loader
            :param target_path: The path to the target data
            :return:            The loaded target
        """
        mat = scipy.io.loadmat(target_path, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        mask = mat["GTcls"].Segmentation
        return Image.fromarray(mask)


@register_dataset(Datasets.PASCAL_VOC_AND_AUG_UNIFIED_DATASET)
class PascalVOCAndAUGUnifiedDataset(ConcatDataset):
    """
    Pascal VOC + AUG train dataset, aka `SBD` dataset contributed in "Semantic contours from inverse detectors".
    This is class implement the common usage of the SBD and PascalVOC datasets as a unified augmented trainset.
    The unified dataset includes a total of 10,582 samples and don't contains duplicate samples from the PascalVOC
    validation set.

    To use this Dataset you need to:

        - Download pascal datasets:
            VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
            AUG 2012: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

        - Unzip and organize it as below:
            pascal_voc_2012
                ├─VOCdevkit
                │ └──VOC2012
                │    ├──JPEGImages
                │    ├──SegmentationClass
                │    ├──ImageSets
                │    │    ├──Segmentation
                │    │    │   └── train.txt
                │    │    ├──Main
                │    │    ├──Action
                │    │    └──Layout
                │    ├──Annotations
                │    └──SegmentationObject
                └──VOCaug
                    ├── aug.txt
                    └── dataset
                          ├──inst
                          ├──img
                          └──cls

        - Instantiate the dataset:
            >> train_set = PascalVOCAndAUGUnifiedDataset(root='.../pascal_voc_2012', ...)

    NOTE: this dataset is only available for training. To test, please use PascalVOC2012SegmentationDataSet.
    """

    def __init__(self, **kwargs):
        print(kwargs)
        if any([kwargs.pop("list_file"), kwargs.pop("samples_sub_directory"), kwargs.pop("targets_sub_directory")]):
            logger.warning(
                "[list_file, samples_sub_directory, targets_sub_directory] arguments passed will not be used"
                " when passed to `PascalVOCAndAUGUnifiedDataset`. Those values are predefined for initiating"
                " the Pascal VOC + AUG training set."
            )
        super().__init__(
            datasets=[
                PascalVOC2012SegmentationDataSet(
                    list_file="VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
                    samples_sub_directory="VOCdevkit/VOC2012/JPEGImages",
                    targets_sub_directory="VOCdevkit/VOC2012/SegmentationClass",
                    **kwargs,
                ),
                PascalAUG2012SegmentationDataSet(
                    list_file="VOCaug/dataset/aug.txt", samples_sub_directory="VOCaug/dataset/img", targets_sub_directory="VOCaug/dataset/cls", **kwargs
                ),
            ]
        )
