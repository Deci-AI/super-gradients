import json
import os

import numpy as np
from PIL import Image

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet


@register_dataset(Datasets.MAPILLARY_DATASET)
class MapillaryDataset(SegmentationDataSet):
    """
    Mapillary Vistas is a large-scale urban street-view dataset.
    This dataset contains 18k, 2k, and 5k images for training, validation and testing with a variety of image
    resolutions, ranging from 1024 × 768 to 4000 × 6000.
    Paper:
        "Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulò, and Peter Kontschieder. The mapillary vistas dataset for
         semantic understanding of street scenes. In CVPR, 2017."
         https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf
    Official site:
        https://www.mapillary.com/ (register for free, then download Vistas dataset)
    """

    """
        Dataset layout:
            root_dir
            ├── config_v1.2.json
            ├── config_v2.0.json
            ├── training
                ├── images
                    ├── {image_name}.jpg            # RGB images
                ├── v1.2
                    ├── labels
                        ├── {image_name}.jpg        # Target masks
                ├── v2.0
                    ├── labels
                        ├── {image_name}.jpg        # Target masks
            ├── validation
            ├── testing
        Note that there are two versions currently available for this dataset, `v1.2` and `v2.0`, the difference according
        to the change log is as follows:
            - Expanded the set of labels to 124 classes (70 instance-specific, 46 stuff, 8 void or crowd).
            - Added raw polygonal annotations as json files. These reflect the ordering in which the segments where
                annotated by the original annotators, i.e. approximately from the background towards the camera.
        The common practice is to use the 65 categorical labels from v1.2 and older.
    """

    IGNORE_LABEL_V1_2 = 65
    IGNORE_LABEL_V2_0 = 123

    def __init__(
        self,
        root_dir: str,
        config_file: str,
        samples_sub_directory: str,
        targets_sub_directory: str,
        sample_extension: str = ".jpg",
        target_extension: str = ".png",
        **kwargs,
    ):
        self.samples_sub_directory = samples_sub_directory
        self.targets_sub_directory = targets_sub_directory
        self.target_extension = target_extension
        self.sample_extension = sample_extension
        # FIXME - Must pass list_file, due to double inheritance error when using DirectoryDataset. See the bug report
        super().__init__(
            root=root_dir,
            samples_sub_directory=samples_sub_directory,
            targets_sub_directory=targets_sub_directory,
            list_file="",
            target_extension=target_extension,
            **kwargs,
        )

        # read in config file
        with open(os.path.join(self.root, config_file), "r") as f:
            config = json.load(f)
        self.labels = config["labels"]
        self.label_colors = [label["color"] for label in self.labels]
        self.label_names = [label["readable"].replace(" ", "_") for label in self.labels]
        # Ignore labels is called `Unlabeled` in config files
        self.ignore_label = self.label_names.index("Unlabeled")
        # SG format requires returning classes as label names without ignore labels, it is also often used to calculate
        # the num of classes.
        self.classes = self.label_names[:-1]

    def _generate_samples_and_targets(self):
        samples_dir = os.path.join(self.root, self.samples_sub_directory)
        labels_dir = os.path.join(self.root, self.targets_sub_directory)

        sample_names = [n for n in sorted(os.listdir(samples_dir)) if n.endswith(self.sample_extension)]
        label_names = [n for n in sorted(os.listdir(labels_dir)) if n.endswith(self.target_extension)]

        assert len(sample_names) == len(label_names), f"Number of samples: {len(sample_names)}," f" doesn't match the number of labels {len(label_names)}"

        for sample_name in sample_names:
            label_path = os.path.join(labels_dir, sample_name.replace(self.sample_extension, self.target_extension))
            sample_path = os.path.join(samples_dir, sample_name)

            if os.path.exists(sample_path) and os.path.exists(label_path):
                self.samples_targets_tuples_list.append((sample_path, label_path))
            else:
                raise AssertionError(f"Sample and/or target file(s) not found or in illegal format " f"(sample path: {sample_path}, target path: {label_path})")

    def apply_color_map(self, target: Image) -> np.array:
        """
        Convert a greyscale target PIL image to an RGB numpy array according to the official Mapillary color map.
        """
        target_array = np.array(target)
        rgb_array = np.zeros((target_array.shape[0], target_array.shape[1], 3), dtype=np.uint8)

        for label_id, color in enumerate(self.label_colors):
            # set all pixels with the current label to the color of the current label
            rgb_array[target_array == label_id] = color

        return rgb_array
