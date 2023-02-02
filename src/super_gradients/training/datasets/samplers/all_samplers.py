from super_gradients.common.object_names import Samplers
from super_gradients.training.datasets.samplers.infinite_sampler import InfiniteSampler
from super_gradients.training.datasets.samplers.repeated_augmentation_sampler import RepeatAugSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler, RandomSampler, WeightedRandomSampler

SAMPLERS = {
    Samplers.INFINITE: InfiniteSampler,
    Samplers.REPEAT_AUG: RepeatAugSampler,
    Samplers.DISTRIBUTED: DistributedSampler,
    Samplers.SEQUENTIAL: SequentialSampler,
    Samplers.SUBSET_RANDOM: SubsetRandomSampler,
    Samplers.RANDOM: RandomSampler,
    Samplers.WEIGHTED_RANDOM: WeightedRandomSampler,
}
