import os
import unittest

import torch
from torch.distributed import init_process_group
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import torch.distributed as dist
from super_gradients import setup_device, init_trainer
from super_gradients.common import MultiGPUMode
from super_gradients.training.datasets.samplers.distributed_sampler_wrapper import DistributedSamplerWrapper


class DummyDataset(Dataset):
    def __init__(self, length=42):
        super().__init__()
        self.length = length

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.length


class DummySampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.num_samples = 2 * len(data_source)

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        return iter(indices * 2)

    def __len__(self):
        return self.num_samples


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class SamplerToDistributedSamplerWrapperTest(unittest.TestCase):
    """
    CallTrainTwiceTest

    Purpose is to call train twice and see nothing crashes. Should be ran with available GPUs (when possible)
    so when calling train again we see there's no change in the model's device.
    """

    def test_dist_sample_wrapper_wraps_non_dist_samplers(self):
        # Create a dummy dataset
        dataset_size = 100
        dataset = DummyDataset(dataset_size)

        # Use DistributedSampler
        sampler = DistributedSampler(dataset)

        # Create a DataLoader with the distributed sampler
        dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

        # Print the indices for each rank
        print(f"Rank {dist.get_rank()}: {list(iter(dataloader))}")


if __name__ == "__main__":
    # setup_device(multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=2)
    unittest.main()
    # dist.destroy_process_group()
