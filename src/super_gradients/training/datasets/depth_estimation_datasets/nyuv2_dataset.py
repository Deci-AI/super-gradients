import warnings

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry import register_dataset
from super_gradients.training.datasets.depth_estimation_datasets.abstract_depth_estimation_dataset import AbstractDepthEstimationDataset
import cv2
import pandas as pd

from super_gradients.training.samples import DepthEstimationSample
import os


@register_dataset(Datasets.NYUV2_DEPTH_ESTIMATION_DATASET)
class NYUv2DepthEstimationDataset(AbstractDepthEstimationDataset):
    """
    Dataset class for NYU Depth V2 dataset for depth estimation.

    :param root: Root directory containing the dataset.
    :param df_path: Path to the CSV file containing image and depth map file paths, relative to root.
    :param transforms: Transforms to be applied to the samples.

    To use the NYUv2Dataset class, ensure that your dataset directory is organized as follows:

    - Root directory (specified as 'root' when initializing the dataset)
      - nyu2_train (or any other split)
        - scene_category_1
          - image_1.jpg
          - image_2.png
          - ...
        - scene_category_2
          - image_1.jpg
          - image_2.png
          - ...
        - ...
      - nyu2_test (or any other split)
        - 00000_colors.png
        - 00001_colors.png
        - 00002_colors.png
        ...

    The CSV file (specified as 'df_path' when initializing the dataset) should contain two columns:
     path to the color images,  path to depth maps (both relative to the root).

    Example CSV content:
    data/nyu2_train/scene_category_1/image_1.jpg,   data/nyu2_train/scene_category_1/image_1_depth.png
    data/nyu2_train/scene_category_1/image_2.jpg,   data/nyu2_train/scene_category_1/image_2_depth.png
    data/nyu2_train/scene_category_2/image_1.jpg,   data/nyu2_train/scene_category_2/image_1_depth.png

    Note: As of 14/12/2023 official downlaod link is broken.
     Data can be obtained at https://www.kaggle.com/code/shreydan/monocular-depth-estimation-nyuv2/input
    ...
    """

    def __init__(self, root: str, df_path: str, transforms=None):
        """
        Initialize NYUv2Dataset.

        :param root: Root directory containing the dataset.
        :param df_path: Path to the CSV file containing image and depth map file paths.
        :param transforms: Transforms to be applied to the samples.
        """
        super(NYUv2DepthEstimationDataset, self).__init__(transforms=transforms)
        self.root = root
        self.df = self._read_df(df_path)
        self._check_paths_exist()

    def _read_df(self, df_path: str) -> pd.DataFrame:
        """
        Read the CSV file containing image and depth map file paths.

        :param df_path: Path to the CSV file.

        :return: DataFrame containing image and depth map file paths.
        """
        df = pd.read_csv(df_path, header=None)
        df[0] = df[0].map(lambda x: os.path.join(self.root, x))
        df[1] = df[1].map(lambda x: os.path.join(self.root, x))
        return df

    def load_sample(self, index: int) -> DepthEstimationSample:
        """
        Load a depth estimation sample at the specified index.

        :param index: Index of the sample.

        :return: Loaded depth estimation sample.
        """
        sample_paths = self.df.iloc[index, :]
        image_path, dp_path = sample_paths[0], sample_paths[1]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        depth_map = cv2.imread(dp_path, cv2.IMREAD_GRAYSCALE)
        return DepthEstimationSample(image=image, depth_map=depth_map)

    def __len__(self):
        """
        Get the number of samples in the dataset.

        :return: Number of samples in the dataset.
        """
        return len(self.df)

    def _check_paths_exist(self):
        """
        Check if the paths in self.df exist. Remove lines with missing paths and print information about removed paths.
        Raise an error if all lines are removed.
        """
        valid_paths = []
        for _, row in self.df.iterrows():
            paths_exist = all(os.path.exists(path) for path in row)
            if paths_exist:
                valid_paths.append(row)
            else:
                warnings.warn(f"Warning: Removed the following line as one or more paths do not exist: {row}")

        if not valid_paths:
            raise FileNotFoundError("All lines in the dataset have been removed as some paths do not exist. " "Please check the paths and dataset structure.")

        self.df = pd.DataFrame(valid_paths, columns=[0, 1])
