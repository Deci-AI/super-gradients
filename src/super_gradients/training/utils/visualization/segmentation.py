import numpy as np
import torch
from torchvision.utils import draw_segmentation_masks
from typing import List, Optional, Tuple, Union, Set
from super_gradients.training.utils.segmentation_utils import to_one_hot
from super_gradients.training.utils.visualization.legend import draw_legend_on_canvas


def overlay_segmentation(
    image: np.ndarray,
    pred_mask: torch.Tensor,
    num_classes: int,
    alpha: float,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw a bounding box on an image.

    :param image:           Image on which to draw the segmentation.
    :param pred_mask:           Image on which to draw the segmentation.
    :param num_classes:           Image on which to draw the segmentation.
    :param alpha:           Float number between [0,1] denoting the transparency of the masks (0 means full transparency, 1 means opacity).
    :param colors:           List containing the colors of the masks or single color for all masks. By default, random colors are generated for each mask.
    :param class_names:           List containing the class names of cityscapes classes used for model training
    """

    overlay = image.copy()
    overlay = torch.from_numpy(overlay.transpose(2, 0, 1))  # torch.from_numpy(overlay.astype(np.uint8)*255)
    segmentation_mask = torch.from_numpy(np.expand_dims(pred_mask.segmentation_map, axis=0))
    one_hot_prediction_masks = to_one_hot(target=segmentation_mask, num_classes=num_classes)  # , ignore_index: int = None)
    segmentation_overlay = draw_segmentation_masks(overlay, masks=one_hot_prediction_masks.squeeze(0).bool(), alpha=alpha, colors=colors)

    segmentation_prediction = np.array(segmentation_overlay.detach().permute(1, 2, 0))

    # Initialize an empty list to store the classes that appear in the image
    classes_in_image_with_color: Set[Tuple[str, Tuple]] = set()

    for idx, class_name in enumerate(class_names):
        color = colors[idx]
        if torch.any(one_hot_prediction_masks[0, idx, :, :]):
            classes_in_image_with_color.add((class_name, color))

    canvas = draw_legend_on_canvas(image=segmentation_prediction, class_color_tuples=classes_in_image_with_color)
    segmentation_prediction = np.concatenate((segmentation_prediction, canvas), axis=0)

    return segmentation_prediction
