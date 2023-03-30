# Copyright (c) Megvii, Inc. and its affiliates.
# Apache 2.0 license: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from deprecate import deprecated

from super_gradients.common.object_names import Samplers
from super_gradients.common.registry.registry import register_sampler


@register_sampler(Samplers.INFINITE)
class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    @deprecated(target=None, deprecated_in="3.0.8", remove_in="3.1.0")
    def __init__(
        self,
        dataset,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        :param size:    Total number of data of the underlying dataset to sample from
        :param shuffle: Whether to shuffle the indices or not
        :param seed:    Initial seed of the shuffle. Must be the same across all workers.
                If None, will use a random seed shared among workers (require synchronization among all workers).
        """
        self._size = len(dataset)
        assert len(dataset) > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size
