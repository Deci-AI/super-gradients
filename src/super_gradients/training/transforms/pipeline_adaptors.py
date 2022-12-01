from typing import Callable
from abc import abstractmethod, ABC
import numpy as np


class TransformsPipelineAdaptorBase(ABC):
    def __init__(self, composed_transforms: Callable):
        self.composed_transforms = composed_transforms

    @abstractmethod
    def __call__(self, sample, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def prep_for_transforms(self, sample):
        raise NotImplementedError

    @abstractmethod
    def post_transforms_processing(self, sample):
        raise NotImplementedError


class AlbumentationsAdaptor(TransformsPipelineAdaptorBase):
    def __init__(self, composed_transforms: Callable):
        super(AlbumentationsAdaptor, self).__init__(composed_transforms)

    def __call__(self, sample, *args, **kwargs):
        sample = self.prep_for_transforms(sample)
        sample = self.composed_transforms(**sample)["image"]
        sample = self.post_transforms_processing(sample)
        return sample

    def prep_for_transforms(self, sample):
        return {"image": np.array(sample)}

    def post_transforms_processing(self, sample):
        return sample
