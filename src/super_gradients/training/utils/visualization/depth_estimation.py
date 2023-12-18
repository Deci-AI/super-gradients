from typing import Optional
import cv2
import numpy as np


class DepthVisualization:
    @staticmethod
    def process_depth_map_for_visualization(
        depth_map: np.ndarray, color_scheme: Optional[int] = None, drop_extreme_percentage: float = 0, inverse: bool = False
    ) -> np.ndarray:
        """
        Process a depth map for visualization.

        This method enhances the visual representation of a depth map by:
        1. Clipping extreme values based on the specified percentage.
        2. Normalizing the depth map to the 0-255 range.
        3. Optionally inverting the depth map (inversion is done as 1 / depth).
        4. Applying a color map using OpenCV's applyColorMap.

        :param depth_map:               Input depth map as a NumPy array.
        :param color_scheme:            OpenCV color scheme for the depth map visualization. If not specified:
                                        - If `inverse=True`, the default is COLORMAP_VIRIDIS.
                                        - If `inverse=False`, the default is COLORMAP_MAGMA.
        :param drop_extreme_percentage: Percentage of extreme values to drop.
        :param inverse:                 Apply inversion (1 / depth) if True.

        :return:                        Processed colormap of the depth map for visualization.
        """
        # Drop extreme values
        if inverse:
            depth_map = 1 / depth_map

        min_val = np.percentile(depth_map, drop_extreme_percentage)
        max_val = np.percentile(depth_map, 100 - drop_extreme_percentage)

        depth_map = np.clip(depth_map, min_val, max_val)

        # Normalize to 0-255
        depth_map_normalized = ((depth_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Determine the default color scheme
        default_color_scheme = cv2.COLORMAP_VIRIDIS if inverse else cv2.COLORMAP_MAGMA

        # Apply colormap
        colormap = cv2.applyColorMap(depth_map_normalized, color_scheme if color_scheme is not None else default_color_scheme)

        # Convert BGR to RGB
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        return colormap_rgb
