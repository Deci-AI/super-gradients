import abc
from typing import List, Optional

import random

from data_gradients.common.decorators import resolve_param
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import Dataset

from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.samples import DepthEstimationSample
from super_gradients.training.transforms.depth_estimation import AbstractDepthEstimationTransform
from super_gradients.training.utils.visualization.depth_estimation import DepthVisualization


class AbstractDepthEstimationDataset(Dataset):
    """
    Abstract class for datasets for depth estimation task.

    Attempting to follow principles provided in pose_etimation_dataset.
    """

    @resolve_param("transforms", ListFactory(TransformsFactory()))
    def __init__(self, transforms: List[AbstractDepthEstimationTransform] = None):
        super().__init__()
        self.transforms = transforms or []

    @abc.abstractmethod
    def load_sample(self, index: int) -> DepthEstimationSample:
        raise NotImplementedError()

    def load_random_sample(self) -> DepthEstimationSample:
        """
        Return a random sample from the dataset

        :return: Instance of DepthEstimationSample
        """
        num_samples = len(self)
        random_index = random.randrange(0, num_samples)
        return self.load_sample(random_index)

    def __getitem__(self, index: int) -> DepthEstimationSample:
        sample = self.load_sample(index)
        for transform in self.transforms:
            sample = transform(sample)
        return sample.image, sample.depth_map

    def plot(
        self,
        max_samples_per_plot: int = 8,
        n_plots: int = 1,
        plot_transformed_data: bool = True,
        color_scheme: Optional[int] = None,
        drop_extreme_percentage: float = 0,
        inverse: bool = False,
    ):
        """
        Combine samples of images with depth maps into plots and display the result.

        :param max_samples_per_plot:    Maximum number of samples (image with depth map) to be displayed per plot.
        :param n_plots:                 Number of plots to display.
        :param plot_transformed_data:   If True, the plot will be over samples after applying transforms (i.e., on __getitem__).
                                        If False, the plot will be over the raw samples (i.e., on load_sample).
        :param color_scheme:            OpenCV color scheme for the depth map visualization. If not specified:
                                        - If `inverse=True`, the default is COLORMAP_VIRIDIS.
                                        - If `inverse=False`, the default is COLORMAP_MAGMA.


        :param drop_extreme_percentage: Percentage of extreme values to drop on both ends of the depth spectrum.
        :param inverse:                 Apply inversion (1 / depth) if True to the depth map.

        :return: None
        """
        plot_counter = 0

        for plot_i in range(n_plots):
            fig, axes = plt.subplots(2, max_samples_per_plot, figsize=(15, 5))
            for img_i in range(max_samples_per_plot):
                index = img_i + plot_i * max_samples_per_plot
                if plot_transformed_data:
                    image, depth_map = self[index]
                else:
                    sample = self.load_sample(index)
                    image, depth_map = sample.image, sample.depth_map

                # Plot the image
                axes[0, img_i].imshow(image)
                axes[0, img_i].axis("off")
                axes[0, img_i].set_title(f"Sample {index}")

                # Plot the depth map side by side with the selected color scheme
                depth_map = DepthVisualization.process_depth_map_for_visualization(depth_map, color_scheme, drop_extreme_percentage, inverse)
                axes[1, img_i].imshow(depth_map)
                axes[1, img_i].axis("off")
                axes[1, img_i].set_title(f"Depth Map {index}")

            plt.show()
            plt.close()

            plot_counter += 1
            if plot_counter == n_plots:
                return
