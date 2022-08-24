from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.samplers.infinite_sampler import InfiniteSampler
from super_gradients.training.datasets.samplers.repeated_augmentation_sampler import RepeatAugSampler
from torch.utils.data.distributed import DistributedSampler


class SamplersFactory(BaseFactory):

    def __init__(self):
        type_dict = {"InfiniteSampler": InfiniteSampler,
                     "RepeatAugSampler": RepeatAugSampler,
                     "DistributedSampler": DistributedSampler
                     }
        super().__init__(type_dict)
