import abc
import random
from typing import Tuple, List, Union

import numpy as np
from omegaconf import ListConfig
from torch.utils.data.dataloader import Dataset

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Processings
from super_gradients.module_interfaces import HasPreprocessingParams
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.transforms.keypoint_transforms import KeypointsCompose, AbstractKeypointTransform
from super_gradients.training.utils.visualization.utils import generate_color_mapping

logger = get_logger(__name__)


class AbstractPoseEstimationDataset(Dataset, HasPreprocessingParams):
    """
    Abstract class for strongly typed dataset classes for pose estimation task.
    This new concept introduced in SG 3.3 and will be used in the future to replace the old BaseKeypointsDataset.
    The reasoning begin strongly typed dataset includes:
    1. Introduction of a new concept of "data sample" that has clear definition (via @dataclass) thus reducing change of bugs/confusion.
    2. Data sample becomes a central concept in data augmentation transforms and metrics.
    3. Dataset implementation decoupled from the model & loss - now the dataset returns the data sample objects
       and model/loss specific conversion happens only in collate function.

    Descendants should implement the load_sample method to read a sample from the disk and return PoseEstimationSample object.
    """

    def __init__(
        self,
        transforms: List[AbstractKeypointTransform],
        num_joints: int,
        edge_links: Union[ListConfig, List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[ListConfig, List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[ListConfig, List[Tuple[int, int, int]], np.ndarray, None],
    ):
        """

        :param transforms: Transforms to be applied to the image & keypoints
        :param num_joints: Number of joints to be predicted
        :param edge_links: Edge links between joints
        :param edge_colors: Color of the edge links. If None, the color will be generated randomly.
        :param keypoint_colors: Color of the keypoints. If None, the color will be generated randomly.
        """
        super().__init__()
        self.transforms = KeypointsCompose(
            transforms,
            load_sample_fn=self.load_random_sample,
        )
        self.num_joints = num_joints

        # Explicitly convert edge_links, keypoint_colors and edge_colors to lists of tuples
        # This is necessary to ensure ListConfig objects do not leak to these properties
        # and from there - to checkpoint's state_dict.
        # Otherwise, through ListConfig instances a whole configuration file will leak to state_dict
        # and torch.load will attempt to unpickle lot of unnecessary classes.
        edge_links = [(int(from_idx), int(to_idx)) for from_idx, to_idx in edge_links]
        if edge_colors is not None:
            edge_colors = [(int(r), int(g), int(b)) for r, g, b in edge_colors]
        if keypoint_colors is not None:
            keypoint_colors = [(int(r), int(g), int(b)) for r, g, b in keypoint_colors]

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
        :return:      Returns an instance of PoseEstimationSample that holds complete sample (image and annotations)
        """
        raise NotImplementedError()

    def load_random_sample(self) -> PoseEstimationSample:
        """
        Return a random sample from the dataset

        :return: Instance of PoseEstimationSample
        """
        num_samples = len(self)
        random_index = random.randrange(0, num_samples)
        return self.load_sample(random_index)

    def __getitem__(self, index: int) -> PoseEstimationSample:
        sample = self.load_sample(index)
        sample = self.transforms.apply_to_sample(sample)
        return sample

    def get_dataset_preprocessing_params(self):
        """

        :return:
        """
        image_to_tensor = {Processings.ImagePermute: {"permutation": (2, 0, 1)}}
        pipeline = self.transforms.get_equivalent_preprocessing() + [image_to_tensor]
        params = dict(
            conf=0.05,
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
        )
        return params
