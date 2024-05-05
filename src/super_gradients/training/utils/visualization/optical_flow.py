import torch
import numpy as np
from torchvision.utils import flow_to_image


class FlowVisualization:
    @staticmethod
    def process_flow_map_for_visualization(
        flow_map: np.ndarray,
    ) -> np.ndarray:
        """
        Process a flow map for visualization.

        :param flow_map:               Input depth map as a NumPy array.

        :return:                       Processed colormap of the flow map for visualization.
        """

        # Convert to Torch tensor
        flow_map = torch.tensor(flow_map)

        # Convert flow map to an image
        flow_map_img = flow_to_image(flow_map)

        # Convert to NumPy array
        flow_map_img = flow_map_img.permute(1, 2, 0).numpy()

        return flow_map_img
