import cv2
import numpy as np


class DepthVisualization:
    @staticmethod
    def process_depth_map_for_visualization(
        depth_map: np.ndarray, color_scheme: int = cv2.COLORMAP_VIRIDIS, drop_extreme_percentage: float = 0, inverse: bool = False
    ) -> np.ndarray:
        """
        Process a depth map for visualization.

        This method enhances the visual representation of a depth map by:
        1. Clipping extreme values based on the specified percentage.
        2. Normalizing the depth map to the 0-255 range.
        3. Optionally inverting the depth map.
        4. Applying a color map using OpenCV's applyColorMap.

        :param depth_map:               Input depth map as a NumPy array.
        :param color_scheme:            OpenCV color scheme for the depth map visualization.
        :param drop_extreme_percentage: Percentage of extreme values to drop.
        :param inverse:                 Apply inversion (255 - depth_map) if True.

        :return:                        Processed colormap of the depth map for visualization.
        """

        # Drop extreme values
        min_val = np.percentile(depth_map, drop_extreme_percentage)
        max_val = np.percentile(depth_map, 100 - drop_extreme_percentage)
        depth_map = np.clip(depth_map, min_val, max_val)

        # Normalize to 0-255
        depth_map_normalized = ((depth_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        if inverse:
            depth_map_normalized = 255 - depth_map_normalized

        # Apply colormap
        colormap = cv2.applyColorMap(depth_map_normalized, color_scheme)
        return colormap
