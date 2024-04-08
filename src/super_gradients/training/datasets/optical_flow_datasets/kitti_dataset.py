import warnings

import numpy as np

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry import register_dataset
from super_gradients.training.datasets.optical_flow_datasets import kitti_utils
from super_gradients.training.datasets.optical_flow_datasets.abstract_optical_flow_dataset import AbstractOpticalFlowDataset

from super_gradients.training.samples import OpticalFlowSample
from glob import glob
import os


@register_dataset(Datasets.KITTI_OPTICAL_FLOW_DATASET)
class KITTIOpticalFlowDataset(AbstractOpticalFlowDataset):
    """
    Dataset class for KITTI 2015 dataset for optical flow.

    :param root: Root directory containing the dataset.
    :param transforms: Transforms to be applied to the samples.

    To use the KITTIOpticalFlowDataset class, ensure that your dataset directory is organized as follows:

    - Root directory (specified as 'root' when initializing the dataset)
      - training
        - image_2
          - 000000_10.png
          - 000000_11.png
          - 000001_10.png
          - 000001_11.png
          - ...
        - flow_occ
          - 000000_10.png
          - 000001_10.png
          - ...

    Data can be obtained at https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow
    ...
    """

    def __init__(self, root: str, transforms=None):
        """
        Initialize KITTIDataset.

        :param root: Root directory containing the dataset.
        :param df_path: Path to the CSV file containing image and depth map file paths.
        :param transforms: Transforms to be applied to the samples.
        """
        super(KITTIOpticalFlowDataset, self).__init__(transforms=transforms)

        images_list = []

        data_root = os.path.join(root, "training")

        images1 = sorted(glob(os.path.join(data_root, "image_2/*_10.png")))
        images2 = sorted(glob(os.path.join(data_root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            images_list += [[img1, img2]]

        flow_list = sorted(glob(os.path.join(data_root, "flow_occ/*_10.png")))

        self.files_list = [(elem1[0], elem1[1], elem2) for elem1, elem2 in zip(images_list, flow_list)]

        self._check_paths_exist()

    def load_sample(self, index: int) -> OpticalFlowSample:
        """
        Load an optical flow estimation sample at the specified index.

        :param index: Index of the sample.

        :return: Loaded optical flow estimation sample.
        """
        flow_map, valid = kitti_utils.read_flow_kitti(self.files_list[index][2])

        image1 = kitti_utils.read_gen(self.files_list[index][0])
        image2 = kitti_utils.read_gen(self.files_list[index][1])

        flow_map = np.array(flow_map).astype(np.float32)
        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        # grayscale images
        if len(image1.shape) == 2:
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        images = np.stack([image1, image2])

        if valid is not None:
            valid = valid
        else:
            valid = (np.abs(flow_map[:, :, 0]) < 1000) & (np.abs(flow_map[:, :, 1]) < 1000)

        return OpticalFlowSample(images=images, flow_map=flow_map, valid=valid)

    def __len__(self):
        """
        Get the number of samples in the dataset.

        :return: Number of samples in the dataset.
        """
        return len(self.files_list)

    def _check_paths_exist(self):
        """
        Check if the paths in self.train_list and self.val_list exist. Remove lines with missing paths and print information about removed paths.
        Raise an error if all lines are removed.
        """
        valid_paths = []

        for idx in range(len(self.files_list)):
            paths_exist = all(os.path.exists(path) for path in self.files_list[idx])
            if paths_exist:
                valid_paths.append(self.files_list[idx])
            else:
                warnings.warn(f"Warning: Removed the following line as one or more paths do not exist: {self.files_list[idx]}")

        if not valid_paths:
            raise FileNotFoundError("All lines in the dataset have been removed as some paths do not exist. " "Please check the paths and dataset structure.")

        self.files_list = valid_paths
