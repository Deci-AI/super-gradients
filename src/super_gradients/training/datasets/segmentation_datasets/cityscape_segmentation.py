import os
import cv2
import numpy as np
from PIL import Image, ImageColor
from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet

# TODO - ADD COARSE DATA - right now cityscapes dataset includes fine annotations. It's optional to use extra coarse
#  annotations.

# label for background and labels to ignore during training and evaluation.
CITYSCAPES_IGNORE_LABEL = 19


class CityscapesDataset(SegmentationDataSet):
    """
    CityscapesDataset - Segmentation Data Set Class for Cityscapes Segmentation Data Set,
    main resolution of dataset: (2048 x 1024).
    Not all the original labels are used for training and evaluation, according to cityscape paper:
    "Classes that are too rare are excluded from our benchmark, leaving 19 classes for evaluation".
    For more details about the dataset labels format see:
    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """

    def __init__(self,
                 root_dir: str,
                 list_file: str,
                 labels_csv_path: str,
                 **kwargs):
        """
        :param root: root directory to dataset.
        :param list_file: list file that contains names of images to load, line format: <image_path> <label_path>
        :param labels_csv_path: path to csv file, with labels metadata and mapping.
        :param kwargs: Any hyper params required for the dataset, i.e img_size, crop_size, cache_images
        """

        self.root_dir = root_dir
        super().__init__(root_dir, list_file=list_file, **kwargs)
        # labels dataframe for labels metadata.
        self.labels_data = np.recfromcsv(os.path.join(self.root_dir, labels_csv_path), dtype='<i8,U20,<i8,<i8,U12,<i8,?,?,U7', comments='&')
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
            self.samples_targets_tuples_list.append((
                os.path.join(self.root, image_path),
                os.path.join(self.root, label_path)
            ))

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
        return Image.fromarray(label, 'L')

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
