from super_gradients.training.datasets.samplers.infinite_sampler import InfiniteSampler
from super_gradients.training.datasets.samplers.repeated_augmentation_sampler import RepeatAugSampler
from torch.utils.data.distributed import DistributedSampler


class SamplerNames:
    INFINITE = "InfiniteSampler"
    REPEAT_AUG = "RepeatAugSampler"
    DISTRIBUTED = "DistributedSampler"


SAMPLERS = {
    SamplerNames.INFINITE: InfiniteSampler,
    SamplerNames.REPEAT_AUG: RepeatAugSampler,
    SamplerNames.DISTRIBUTED: DistributedSampler
 }
