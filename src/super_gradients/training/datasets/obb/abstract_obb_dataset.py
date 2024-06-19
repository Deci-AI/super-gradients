import abc
from abc import ABC

from super_gradients.training.transforms.obb import OBBSample
from torch.utils.data import Dataset


class AbstractOBBDataset(Dataset, ABC):
    """
    Abstract class for OBB detection datasets.
    This class declares minimal interface for OBB detection datasets.
    """

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index) -> OBBSample:
        pass
