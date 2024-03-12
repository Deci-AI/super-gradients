from abc import ABC, abstractmethod
import numpy as np


class HasClassesInformation(ABC):
    @abstractmethod
    def get_sample_classes_information(self, index) -> np.ndarray:
        """
        Returns a histogram of length `num_classes` with class occurrences at that index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataset_classes_information(self) -> np.ndarray:
        """
        Returns a matrix of shape (dataset_length, num_classes). Each row `i` is histogram of length `num_classes` with class occurrences for sample `i`.
        Example implementation, assuming __len__: `np.vstack([self.get_sample_classes_information(i) for i in range(len(self))])`
        """
        raise NotImplementedError
