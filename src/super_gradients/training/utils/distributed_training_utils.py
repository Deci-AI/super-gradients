import itertools
from contextlib import contextmanager
import torch
from torch.cuda.amp import autocast
import torch.nn as nn


@contextmanager
def run_once(timeout=None, group=None):
    """
    This context manager should be used to wrap code that should run only on rank-0.

    In addition to checking the rank, this method correctly implements a barrier with an optional timeout to make sure rank-0 finishes it's job
        before continuing.

    :param timeout: optional timeout in seconds for the secondary processes to wait for rank-0 to finish it's job.
    :enter should_run: if True, the process entering the context manager is the one that should do the work.

    Note: since python's context-manager protocol doesn't allow for "optionally yielding" (entering the context), we must enter with a boolean value
        indicating whether the process should do any work, and can't implement this "if" inside run_once.

    Example:
    ```
    with run_once() as should_run:
        if should_run:
            do_work()

    @run_once()
    def this_works_as_well(should_run):
        if not should_run:
            return
        do_work()
    ```
    """
    # NOT DDP - WE NEED TO RUN THE JOB, BUT DON'T NEED TO SYNC
    if not torch.distributed.is_initialized():
        should_run = True
        should_sync = False

    # DDP - RUN ONLY ON RANK 0 AND SYNC
    else:
        should_run = (torch.distributed.get_rank() == 0)
        should_sync = True

    yield should_run

    if should_sync:
        torch.distributed.monitored_barrier(group=group, timeout=timeout)


def distributed_all_reduce_tensor_average(tensor, n):
    """
    This method performs a reduce operation on multiple nodes running distributed training
    It first sums all of the results and then divides the summation
    :param tensor:  The tensor to perform the reduce operation for
    :param n:  Number of nodes
    :return:   Averaged tensor from all of the nodes
    """
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n
    return rt


def reduce_results_tuple_for_ddp(validation_results_tuple, device):
    """Gather all validation tuples from the various devices and average them"""
    validation_results_list = list(validation_results_tuple)
    for i, validation_result in enumerate(validation_results_list):
        validation_results_list[i] = distributed_all_reduce_tensor_average(torch.tensor(validation_result).to(device),
                                                                           torch.distributed.get_world_size())
    validation_results_tuple = tuple(validation_results_list)
    return validation_results_tuple


class MultiGPUModeAutocastWrapper():
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        with autocast():
            out = self.func(*args, **kwargs)
        return out


def scaled_all_reduce(tensors: torch.Tensor, num_gpus: int):
    """
    Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place.
    Currently supports only the sum
    reduction operator.
    The reduced values are scaled by the inverse size of the
    process group (equivalent to num_gpus).
    """
    # There is no need for reduction in the single-proc case
    if num_gpus == 1:
        return tensors

    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)

    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()

    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / num_gpus)
    return tensors


@torch.no_grad()
def compute_precise_bn_stats(model: nn.Module, loader: torch.utils.data.DataLoader, precise_bn_batch_size: int, num_gpus: int):
    '''
    :param model:                   The model being trained (ie: SgModel.net)
    :param loader:                  Training dataloader (ie: SgModel.train_loader)
    :param precise_bn_batch_size:   The effective batch size we want to calculate the batchnorm on. For example, if we are training a model
                                    on 8 gpus, with a batch of 128 on each gpu, a good rule of thumb would be to give it 8192
                                    (ie: effective_batch_size * num_gpus = batch_per_gpu * num_gpus * num_gpus).
                                    If precise_bn_batch_size is not provided in the training_params, the latter heuristic
                                    will be taken.
    param num_gpus:                 The number of gpus we are training on
    '''

    # Compute the number of minibatches to use
    num_iter = int(precise_bn_batch_size / (loader.batch_size * num_gpus)) if precise_bn_batch_size else num_gpus
    num_iter = min(num_iter, len(loader))

    # Retrieve the BN layers
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]

    # Initialize BN stats storage for computing mean(mean(batch)) and mean(var(batch))
    running_means = [torch.zeros_like(bn.running_mean) for bn in bns]
    running_vars = [torch.zeros_like(bn.running_var) for bn in bns]

    # Remember momentum values
    momentums = [bn.momentum for bn in bns]

    # Set momentum to 1.0 to compute BN stats that only reflect the current batch
    for bn in bns:
        bn.momentum = 1.0

    # Average the BN stats for each BN layer over the batches
    for inputs, _labels in itertools.islice(loader, num_iter):
        model(inputs.cuda())
        for i, bn in enumerate(bns):
            running_means[i] += bn.running_mean / num_iter
            running_vars[i] += bn.running_var / num_iter

    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = scaled_all_reduce(running_means, num_gpus=num_gpus)
    running_vars = scaled_all_reduce(running_vars, num_gpus=num_gpus)

    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]
