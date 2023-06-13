import abc
from typing import Tuple, List, Mapping, Any, Dict, Union

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate, Dataset

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Processings
from super_gradients.common.registry.registry import register_collate_function
from super_gradients.training.datasets.pose_estimation_datasets.target_generators import KeypointsTargetsGenerator
from super_gradients.training.transforms.keypoint_transforms import KeypointsCompose, KeypointTransform
from super_gradients.training.utils.visualization.utils import generate_color_mapping

logger = get_logger(__name__)


class BaseKeypointsDataset(Dataset):
    """
    Base class for pose estimation datasets.
    Descendants should implement the load_sample method to read a sample from the disk and return (image, mask, joints, extras) tuple.
    """

    def __init__(
        self,
        target_generator: KeypointsTargetsGenerator,
        transforms: List[KeypointTransform],
        min_instance_area: float,
        num_joints: int,
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
    ):
        """

        :param target_generator: Target generator that will be used to generate the targets for the model.
            See DEKRTargetsGenerator for an example.
        :param transforms: Transforms to be applied to the image & keypoints
        :param min_instance_area: Minimum area of an instance to be included in the dataset
        :param num_joints: Number of joints to be predicted
        :param edge_links: Edge links between joints
        :param edge_colors: Color of the edge links. If None, the color will be generated randomly.
        :param keypoint_colors: Color of the keypoints. If None, the color will be generated randomly.
        """
        super().__init__()
        self.target_generator = target_generator
        self.transforms = KeypointsCompose(transforms)
        self.min_instance_area = min_instance_area
        self.num_joints = num_joints
        self.edge_links = edge_links
        self.edge_colors = edge_colors or generate_color_mapping(len(edge_links))
        self.keypoint_colors = keypoint_colors or generate_color_mapping(num_joints)

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_sample(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Read a sample from the disk and return (image, mask, joints, extras) tuple
        :param index: Sample index
        :return: Tuple of (image, mask, joints, extras)
            image - Numpy array of [H,W,3] shape, which represents input RGB image
            mask - Numpy array of [H,W] shape, which represents a binary mask with zero values corresponding to an
                    ignored region which should not be used for training (contribute to loss)
            joints - Numpy array of [Num Instances, Num Joints, 3] shape, which represents the skeletons of the instances
            extras - Dictionary of extra information about the sample that should be included in `extras` dictionary.
        """
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any, Mapping[str, Any]]:
        img, mask, joints, extras = self.load_sample(index)
        img, mask, joints, _, _ = self.transforms(img, mask, joints, areas=None, bboxes=None)

        joints = self.filter_joints(joints, img)

        targets = self.target_generator(img, joints, mask)
        return img, targets, {"gt_joints": joints, **extras}

    def compute_area(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute area of a bounding box for each instance.
        :param joints:  [Num Instances, Num Joints, 3]
        :return: [Num Instances]
        """
        w = np.max(joints[:, :, 0], axis=-1) - np.min(joints[:, :, 0], axis=-1)
        h = np.max(joints[:, :, 1], axis=-1) - np.min(joints[:, :, 1], axis=-1)
        return w * h

    def filter_joints(self, joints: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Filter instances that are either too small or do not have visible keypoints
        :param joints: Array of shape [Num Instances, Num Joints, 3]
        :param image:
        :return: [New Num Instances, Num Joints, 3], New Num Instances <= Num Instances
        """
        # Update visibility of joints for those that are outside the image
        outside_image_mask = (joints[:, :, 0] < 0) | (joints[:, :, 1] < 0) | (joints[:, :, 0] >= image.shape[1]) | (joints[:, :, 1] >= image.shape[0])
        joints[outside_image_mask, 2] = 0

        # Filter instances with all invisible keypoints
        instances_with_visible_joints = np.count_nonzero(joints[:, :, 2], axis=-1) > 0
        joints = joints[instances_with_visible_joints]

        # Remove instances with too small area
        areas = self.compute_area(joints)
        joints = joints[areas > self.min_instance_area]

        return joints

    def get_dataset_preprocessing_params(self):
        """

        :return:
        """
        pipeline = self.transforms.get_equivalent_preprocessing()
        params = dict(
            conf=0.25,
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
        )
        return params


@register_collate_function()
class KeypointsCollate:
    """
    Collate image & targets, return extras as is.
    """

    def __call__(self, batch):
        images = []
        targets = []
        extras = []
        for image, target, extra in batch:
            images.append(image)
            targets.append(target)
            extras.append(extra)

        extras = {k: [dic[k] for dic in extras] for k in extras[0]}  # Convert list of dicts to dict of lists

        images = default_collate(images)
        targets = default_collate(targets)
        return images, targets, extras
