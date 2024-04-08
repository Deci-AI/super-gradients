import abc
from typing import List, Tuple

import random

import numpy as np
from data_gradients.common.decorators import resolve_param
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import Dataset

from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.samples import OpticalFlowSample
from super_gradients.training.transforms.optical_flow import AbstractOpticalFlowTransform
from super_gradients.training.utils.visualization.optical_flow import FlowVisualization


class AbstractOpticalFlowDataset(Dataset):
    """
    Abstract class for datasets for optical flow task.

    Attempting to follow principles provided in pose_etimation_dataset.
    """

    @resolve_param("transforms", ListFactory(TransformsFactory()))
    def __init__(self, transforms: List[AbstractOpticalFlowTransform] = None):
        super().__init__()
        self.transforms = transforms or []

    @abc.abstractmethod
    def load_sample(self, index: int) -> OpticalFlowSample:
        """
        Load an optical flow sample from the dataset.

        :param index: Index of the sample to load.
        :return: Instance of OpticalFlowSample.

        """
        raise NotImplementedError()

    def load_random_sample(self) -> OpticalFlowSample:
        """
        Return a random sample from the dataset

        :return: Instance of OpticalFlowSample
        """
        num_samples = len(self)
        random_index = random.randrange(0, num_samples)
        return self.load_sample(random_index)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a transformed optical flow sample from the dataset.

        :param index: Index of the sample to retrieve.
        :return: Tuple containing the transformed images and flow map as np.ndarrays.

        After applying the transforms pipeline, the image is expected to be in 2HWC format, and the flow map should be
        a 3D array (e.g., 2 x Height x Width).

        Before returning the images and flow map, the image's channels are moved to 2CHW format and the flow_map's channels are moved to CHW format.
        """
        sample = self.load_sample(index)
        for transform in self.transforms:
            sample = transform(sample)
        return np.transpose(sample.images, (0, 3, 1, 2)).astype(np.float32), (
            np.transpose(sample.flow_map, (2, 0, 1)).astype(np.float32),
            sample.valid.astype(np.float32),
        )

    def plot(
        self,
        max_samples_per_plot: int = 8,
        n_plots: int = 1,
        plot_transformed_data: bool = True,
    ):
        """
        Combine samples of images with flow maps into plots and display the result.

        :param max_samples_per_plot:    Maximum number of samples (image with depth map) to be displayed per plot.
        :param n_plots:                 Number of plots to display.
        :param plot_transformed_data:   If True, the plot will be over samples after applying transforms (i.e., on __getitem__).
                                        If False, the plot will be over the raw samples (i.e., on load_sample).

        :return: None
        """
        plot_counter = 0

        for plot_i in range(n_plots):
            fig, axes = plt.subplots(3, max_samples_per_plot, figsize=(20, 7))
            for sample_i in range(max_samples_per_plot):
                index = sample_i + plot_i * max_samples_per_plot
                if plot_transformed_data:
                    images, (flow_map, valid) = self[index]

                    # Transpose to HWC format for visualization
                    images = images.transpose(0, 2, 3, 1)
                    flow_map = flow_map.squeeze()  # Remove dummy dimension
                else:
                    sample = self.load_sample(index)
                    images, flow_map, _ = sample.images, sample.flow_map.sample.valid

                # Plot the image
                axes[0, sample_i].imshow(images[0].astype(np.uint8))
                axes[0, sample_i].axis("off")
                axes[0, sample_i].set_title(f"Sample {index} image1")

                axes[1, sample_i].imshow(images[1].astype(np.uint8))
                axes[1, sample_i].axis("off")
                axes[1, sample_i].set_title(f"Sample {index} image2")

                # Plot the depth map side by side with the selected color scheme
                flow_map = FlowVisualization.process_flow_map_for_visualization(flow_map)
                axes[2, sample_i].imshow(flow_map)
                axes[2, sample_i].axis("off")
                axes[2, sample_i].set_title(f"Flow Map {index}")

            plt.show()
            plt.close()

            plot_counter += 1
            if plot_counter == n_plots:
                return
