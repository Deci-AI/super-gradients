import os

import numpy as np

from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet

PASCAL_VOC_2012_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
    'tv/monitor', 'ambigious'
]


class PascalVOC2012SegmentationDataSet(SegmentationDataSet):
    """
    PascalVOC2012SegmentationDataSet - Segmentation Data Set Class for Pascal VOC 2012 Data Set
    """

    def __init__(self, sample_suffix=None, target_suffix=None, *args, **kwargs):
        self.sample_suffix = '.jpg' if sample_suffix is None else sample_suffix
        self.target_suffix = '.png' if target_suffix is None else target_suffix
        super().__init__(*args, **kwargs)

        # THERE ARE 21 CLASSES, AND BACKGROUND
        self.classes = PASCAL_VOC_2012_CLASSES

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

        # REMOVING THE BACKGROUND CLASS FROM THE PLOTS
        num_classes_to_plot = len(self.classes) - 1
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
                image_path = os.path.join(self.root, self.samples_sub_directory, line.rstrip('\n') + self.sample_suffix)
                mask_path = os.path.join(self.root, self.targets_sub_directory, line.rstrip('\n') + self.target_suffix)

                if os.path.exists(mask_path) and os.path.exists(image_path):
                    self.samples_targets_tuples_list.append((image_path, mask_path))

        # GENERATE SAMPLES AND TARGETS OF THE SEGMENTATION DATA SET CLASS
        super()._generate_samples_and_targets()

    def _get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
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
