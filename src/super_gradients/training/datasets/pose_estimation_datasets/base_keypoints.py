import abc
import random
from typing import Tuple, List, Mapping, Any, Union

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate, Dataset

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Processings
from super_gradients.common.registry.registry import register_collate_function
from super_gradients.module_interfaces import HasPreprocessingParams
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.datasets.pose_estimation_datasets.target_generators import KeypointsTargetsGenerator
from super_gradients.training.transforms.keypoint_transforms import KeypointsCompose, AbstractKeypointTransform
from super_gradients.training.utils.visualization.utils import generate_color_mapping

logger = get_logger(__name__)


class BaseKeypointsDataset(Dataset, HasPreprocessingParams):
    """
    Base class for pose estimation datasets.
    Descendants should implement the load_sample method to read a sample from the disk and return (image, mask, joints, extras) tuple.
    """

    def __init__(
        self,
        target_generator: KeypointsTargetsGenerator,
        transforms: List[AbstractKeypointTransform],
        min_instance_area: float,
        num_joints: int,
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
    ):
        """

        :param target_generator: Target generator that will be used to generate the targets for the model.
        :param transforms: Transforms to be applied to the image & keypoints
        :param min_instance_area: Minimum area of an instance to be included in the dataset
        :param num_joints: Number of joints to be predicted
        :param edge_links: Edge links between joints
        :param edge_colors: Color of the edge links. If None, the color will be generated randomly.
        :param keypoint_colors: Color of the keypoints. If None, the color will be generated randomly.
        """
        super().__init__()
        self.target_generator = target_generator
        self.transforms = KeypointsCompose(transforms, load_sample_fn=self.load_random_sample, min_bbox_area=min_instance_area, min_visible_joints=1)
        self.num_joints = num_joints
        self.edge_links = edge_links
        self.edge_colors = edge_colors or generate_color_mapping(len(edge_links))
        self.keypoint_colors = keypoint_colors or generate_color_mapping(num_joints)

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_sample(self, index: int) -> PoseEstimationSample:
        """
        Read a sample from the disk and return a PoseEstimationSample
        :param index: Sample index
        :return: Returns an instance of PoseEstimationSample that holds complete sample (image and annotations)
        """
        raise NotImplementedError()

    def load_random_sample(self):
        num_samples = len(self)
        random_index = random.randint(0, num_samples)
        return self.load_sample(random_index)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any, Mapping[str, Any]]:
        sample = self.load_sample(index)
        sample = self.transforms(sample)

        targets = self.target_generator(sample)
        image = sample.image
        sample.image = None  # This is to save memory & time and not pass image tensor second time from dataloader
        return image, targets, {"groundtruth_samples": sample}

    def get_dataset_preprocessing_params(self):
        """

        :return:
        """
        pipeline = self.transforms.get_equivalent_preprocessing()
        params = dict(
            conf=0.05,
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
