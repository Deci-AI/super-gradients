import sys
import itertools
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.cuda.amp import autocast
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.common.environment.env_helpers import find_free_port, is_distributed
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment import environment_config

logger = get_logger(__name__)


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
        if torch.is_tensor(validation_result):
            validation_result = validation_result.clone().detach()
        else:
            validation_result = torch.tensor(validation_result)
        validation_results_list[i] = distributed_all_reduce_tensor_average(tensor=validation_result.to(device), n=torch.distributed.get_world_size())
    validation_results_tuple = tuple(validation_results_list)
    return validation_results_tuple


class MultiGPUModeAutocastWrapper:
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
    """
    :param model:                   The model being trained (ie: Trainer.net)
    :param loader:                  Training dataloader (ie: Trainer.train_loader)
    :param precise_bn_batch_size:   The effective batch size we want to calculate the batchnorm on. For example, if we are training a model
                                    on 8 gpus, with a batch of 128 on each gpu, a good rule of thumb would be to give it 8192
                                    (ie: effective_batch_size * num_gpus = batch_per_gpu * num_gpus * num_gpus).
                                    If precise_bn_batch_size is not provided in the training_params, the latter heuristic
                                    will be taken.
    param num_gpus:                 The number of gpus we are training on
    """

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


def get_local_rank():
    """
    Returns the local rank if running in DDP, and 0 otherwise
    :return: local rank
    """
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """
    Returns the world size if running in DDP, and 1 otherwise
    :return: world size
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@contextmanager
def wait_for_the_master(local_rank: int):
    """
    Make all processes waiting for the master to do some task.
    """
    if local_rank > 0:
        dist.barrier()
    yield
    if local_rank == 0:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        else:
            dist.barrier()


def setup_gpu_mode(gpu_mode: MultiGPUMode = MultiGPUMode.OFF, num_gpus: int = None):
    """
    If required, launch ddp subprocesses.
    :param gpu_mode:    DDP, DP or Off
    :param num_gpus:    Number of GPU's to use.
    """
    if gpu_mode == MultiGPUMode.AUTO and torch.cuda.device_count() > 1:
        gpu_mode = MultiGPUMode.DISTRIBUTED_DATA_PARALLEL
    if require_gpu_setup(gpu_mode):
        num_gpus = num_gpus or torch.cuda.device_count()
        if num_gpus > torch.cuda.device_count():
            raise ValueError(f"You specified num_gpus={num_gpus} but only {torch.cuda.device_count()} GPU's are available")
        restart_script_with_ddp(num_gpus)


def require_gpu_setup(gpu_mode: MultiGPUMode) -> bool:
    """Check if the environment requires a setup in order to work with DDP."""
    return (gpu_mode == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL) and (not is_distributed())


@record
def restart_script_with_ddp(num_gpus: int = None):
    """Launch the same script as the one that was launched (i.e. the command used to start the current process is re-used) but on subprocesses (i.e. with DDP).

    :param num_gpus: How many gpu's you want to run the script on. If not specified, every available device will be used.
    """
    ddp_port = find_free_port()

    # Get the value fom recipe if specified, otherwise take all available devices.
    num_gpus = num_gpus if num_gpus else torch.cuda.device_count()
    if num_gpus > torch.cuda.device_count():
        raise ValueError(f"You specified num_gpus={num_gpus} but only {torch.cuda.device_count()} GPU's are available")

    logger.info(
        "Launching DDP with:\n"
        f"   - ddp_port = {ddp_port}\n"
        f"   - num_gpus = {num_gpus}/{torch.cuda.device_count()} available\n"
        "-------------------------------------\n"
    )

    config = LaunchConfig(
        nproc_per_node=num_gpus,
        min_nodes=1,
        max_nodes=1,
        run_id="none",
        role="default",
        rdzv_endpoint=f"127.0.0.1:{ddp_port}",
        rdzv_backend="static",
        rdzv_configs={"rank": 0, "timeout": 900},
        rdzv_timeout=-1,
        max_restarts=0,
        monitor_interval=5,
        start_method="spawn",
        log_dir=None,
        redirects=Std.NONE,
        tee=Std.NONE,
        metrics_cfg={},
    )

    elastic_launch(config=config, entrypoint=sys.executable)(*sys.argv, *environment_config.EXTRA_ARGS)

    # The code below should actually never be reached as the process will be in a loop inside elastic_launch until any subprocess crashes.
    sys.exit("Main process finished")


def get_gpu_mem_utilization():
    """GPU memory managed by the caching allocator in bytes for a given device."""

    # Workaround to work on any torch version
    if hasattr(torch.cuda, "memory_reserved"):
        return torch.cuda.memory_reserved()
    else:
        return torch.cuda.memory_cached()
