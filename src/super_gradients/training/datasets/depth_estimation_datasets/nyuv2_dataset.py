from super_gradients.training.datasets.depth_estimation_datasets.abstract_depth_estimation_dataset import AbstractDepthEstimationDataset
import cv2
import pandas as pd

from super_gradients.training.samples import DepthEstimationSample
import os


class NYUv2Dataset(AbstractDepthEstimationDataset):
    def __init__(self, root: str, df_path, transforms=None):
        super(NYUv2Dataset, self).__init__(transforms=transforms)
        self.root = root
        self.df = self._read_df(df_path)

    def _read_df(self, df_path: str) -> pd.DataFrame:
        df = pd.read_csv(df_path, header=None)
        df[0] = df[0].map(lambda x: os.path.join(self.root, x))
        df[1] = df[1].map(lambda x: os.path.join(self.root, x))
        return df

    def load_sample(self, index: int) -> DepthEstimationSample:
        sample_paths = self.df.iloc[index, :]
        image_path, dp_path = sample_paths[0], sample_paths[1]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        depth_map = cv2.imread(dp_path, cv2.IMREAD_GRAYSCALE)
        return DepthEstimationSample(image=image, depth_map=depth_map)

    def _open_im(self, p, gray=False):
        im = cv2.imread(str(p))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB)
        return im

    def __len__(
        self,
    ):
        return len(self.df)
