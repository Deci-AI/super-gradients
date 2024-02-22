import collections
import sys

import torch
from torch.utils.data import Dataset, DataLoader

from super_gradients import setup_device
from super_gradients.training.datasets.samplers.distributed_sampler_wrapper import DistributedSamplerWrapper


class DummyDataset(Dataset):
    def __init__(self, length=42):
        super().__init__()
        self.length = length

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.length


class RepeatSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, repeat_times):
        self.data_source = data_source
        self.repeat_times = repeat_times
        self.num_samples = repeat_times * len(data_source)

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        return iter(indices * self.repeat_times)

    def __len__(self):
        return self.num_samples


def aggregate_epoch(data_loader):
    results = list()

    for batch in data_loader:
        for element in batch:
            results.append(element)
    return results


def compare_counts(x, y):
    return collections.Counter(x) == collections.Counter(y)


if __name__ == "__main__":
    n_gpus = 2
    sampler_n_repeats = 3
    bs = 4
    data_size = 10 * n_gpus * bs

    setup_device(
        device="cuda",
        multi_gpu="DDP",
        num_gpus=n_gpus,
    )

    dataset = DummyDataset(length=data_size)
    sampler = RepeatSampler(dataset, repeat_times=sampler_n_repeats)
    dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler)

    # Test *non-distributed* sampler *in DDP mode*
    # THIS IS BAD EXAMPLE BECAUSE YOU EXPECT A DISTRIBUTED SAMPLER TO BE USED IN DDP MODE
    # The expected `len(dataloader)` when implemented correctly should ALSO be divided by `n_gpus`
    if len(dataloader) != (data_size * sampler_n_repeats) / bs:
        print(f"Wrong DataLoader length. Expected: {((data_size * sampler_n_repeats) / bs)=}, got {len(dataloader)}")
        torch.distributed.destroy_process_group()
        sys.exit(1)

    epoch_data = aggregate_epoch(dataloader)
    if not compare_counts(epoch_data, [i for i in dataset] * sampler_n_repeats * n_gpus):  # Note that epoch contains `n_gpus`-times more elements.
        torch.distributed.destroy_process_group()
        sys.exit(1)

    dist_sampler = DistributedSamplerWrapper(sampler)
    dataloader = DataLoader(dataset, batch_size=bs, sampler=dist_sampler)

    if len(dataloader) != (data_size * sampler_n_repeats) / (bs * n_gpus):
        print(f"Wrong DataLoader length. Expected: {((data_size * sampler_n_repeats) / (bs*n_gpus))=}, got {len(dataloader)}")
        torch.distributed.destroy_process_group()
        sys.exit(1)

    epoch_data = aggregate_epoch(dataloader)
    if not compare_counts(epoch_data, [i for i in dataset] * sampler_n_repeats):
        torch.distributed.destroy_process_group()
        sys.exit(1)

    torch.distributed.destroy_process_group()
    sys.exit(0)
