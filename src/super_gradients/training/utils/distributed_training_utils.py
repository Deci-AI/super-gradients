import sys
import os
import itertools
from typing import List, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.cuda.amp import autocast
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from super_gradients.common.environment.ddp_utils import init_trainer
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.common.environment.argparse_utils import EXTRA_ARGS
from super_gradients.common.environment.ddp_utils import find_free_port, is_distributed, is_launched_using_sg


from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.abstractions.mute_processes import mute_current_process
from super_gradients.common.environment.device_utils import device_config

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.type_factory import TypeFactory

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


def require_ddp_setup() -> bool:
    return device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL and device_config.assigned_rank != get_local_rank()


def is_ddp_subprocess():
    return torch.distributed.get_rank() > 0 if dist.is_initialized() else False


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


def get_device_ids() -> List[int]:
    return list(range(get_world_size()))


def count_used_devices() -> int:
    return len(get_device_ids())


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
    """[DEPRECATED in favor of setup_device] If required, launch ddp subprocesses.
    :param gpu_mode:    DDP, DP, Off or AUTO
    :param num_gpus:    Number of GPU's to use. When None, use all available devices on DDP or only one device on DP/OFF.
    """
    logger.warning("setup_gpu_mode is now deprecated in favor of setup_device")
    setup_device(multi_gpu=gpu_mode, num_gpus=num_gpus)


@resolve_param("multi_gpu", TypeFactory(MultiGPUMode.dict()))
def setup_device(multi_gpu: MultiGPUMode = MultiGPUMode.AUTO, num_gpus: int = None, device: str = "cuda"):
    """
    If required, launch ddp subprocesses.
    :param multi_gpu:   DDP, DP, Off or AUTO
    :param num_gpus:    Number of GPU's to use. When None, use all available devices on DDP or only one device on DP/OFF.
    :param device:      The device you want to use ('cpu' or 'cuda')

    If you only set num_gpus, your device will be set up according to the following logic:
        - `setup_device(num_gpus=0)`  => `gpu_mode='OFF'` and `device='cpu'`
        - `setup_device(num_gpus=1)`  => `gpu_mode='OFF'` and `device='gpu'`
        - `setup_device(num_gpus>=2)` => `gpu_mode='DDP'` and `device='gpu'`
        - `setup_device(num_gpus=-1)` => `gpu_mode='DDP'` and `device='gpu'` and `num_gpus=<N-AVAILABLE-GPUs>`

    """
    init_trainer()

    # When launching with torch.distributed.launch or torchrun, multi_gpu might not be set to DDP (since we are not using the recipe params)
    # To avoid any issue we force multi_gpu to be DDP if the current process is ddp subprocess. We also set num_gpus, device to run smoothly.
    if not is_launched_using_sg() and is_distributed():
        multi_gpu, num_gpus, device = MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, None, "cuda"

    if device is None:
        device = "cuda"

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA device is not available on your device... Moving to CPU.")
        multi_gpu, num_gpus, device = MultiGPUMode.OFF, 0, "cpu"

    if device == "cpu":
        setup_cpu(multi_gpu, num_gpus)
    elif device == "cuda":
        setup_gpu(multi_gpu, num_gpus)
    else:
        raise ValueError(f"Only valid values for device are: 'cpu' and 'cuda'. Received: '{device}'")


def setup_cpu(multi_gpu: MultiGPUMode = MultiGPUMode.AUTO, num_gpus: int = None):
    """
    :param multi_gpu:    DDP, DP, Off or AUTO
    :param num_gpus:     Number of GPU's to use.
    """
    if multi_gpu not in (MultiGPUMode.OFF, MultiGPUMode.AUTO):
        raise ValueError(f"device='cpu' and multi_gpu={multi_gpu} are not compatible together.")

    if num_gpus not in (0, None):
        raise ValueError(f"device='cpu' and num_gpus={num_gpus} are not compatible together.")

    device_config.device = "cpu"
    device_config.multi_gpu = MultiGPUMode.OFF


def setup_gpu(multi_gpu: MultiGPUMode = MultiGPUMode.AUTO, num_gpus: int = None):
    """
    If required, launch ddp subprocesses.
    :param multi_gpu:    DDP, DP, Off or AUTO
    :param num_gpus:     Number of GPU's to use. When None, use all available devices on DDP or only one device on DP/OFF.
    """

    if num_gpus == 0:
        raise ValueError("device='cuda' and num_gpus=0 are not compatible together.")

    multi_gpu, num_gpus = _resolve_gpu_params(multi_gpu=multi_gpu, num_gpus=num_gpus)

    device_config.device = "cuda"
    device_config.multi_gpu = multi_gpu

    if is_distributed():
        initialize_ddp()
    elif multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
        restart_script_with_ddp(num_gpus=num_gpus)


def _resolve_gpu_params(multi_gpu: MultiGPUMode, num_gpus: int) -> Tuple[MultiGPUMode, int]:
    """
    Resolve the values multi_gpu in (None, MultiGPUMode.AUTO) and num_gpus in (None, -1), and check compatibility between both parameters.
    :param multi_gpu:    DDP, DP, Off or AUTO
    :param num_gpus:     Number of GPU's to use. When None, use all available devices on DDP or only one device on DP/OFF.
    """

    # Resolve None
    if multi_gpu is None:
        if num_gpus is None:  # When Nothing is specified, just run on single GPU
            multi_gpu = MultiGPUMode.OFF
            num_gpus = 1
        else:
            multi_gpu = MultiGPUMode.AUTO

    if num_gpus is None:
        num_gpus = -1

    # Resolve multi_gpu
    if num_gpus == -1:
        if multi_gpu in (MultiGPUMode.OFF, MultiGPUMode.DATA_PARALLEL):
            num_gpus = 1
        elif multi_gpu in (MultiGPUMode.AUTO, MultiGPUMode.DISTRIBUTED_DATA_PARALLEL):
            num_gpus = torch.cuda.device_count()

    # Resolve multi_gpu
    if multi_gpu == MultiGPUMode.AUTO:
        if num_gpus > 1:
            multi_gpu = MultiGPUMode.DISTRIBUTED_DATA_PARALLEL
        else:
            multi_gpu = MultiGPUMode.OFF

    # Check compatibility between num_gpus and multi_gpu
    if multi_gpu in (MultiGPUMode.OFF, MultiGPUMode.DATA_PARALLEL):
        if num_gpus != 1:
            raise ValueError(f"You specified num_gpus={num_gpus} but it has not be 1 on when working with multi_gpu={multi_gpu}")
    else:
        if num_gpus > torch.cuda.device_count():
            raise ValueError(f"You specified num_gpus={num_gpus} but only {torch.cuda.device_count()} GPU's are available")
    return multi_gpu, num_gpus


def initialize_ddp():
    """
    Initialize Distributed Data Parallel

    Important note: (1) in distributed training it is customary to specify learning rates and batch sizes per GPU.
    Whatever learning rate and schedule you specify will be applied to the each GPU individually.
    Since gradients are passed and summed (reduced) from all to all GPUs, the effective batch size is the
    batch you specify times the number of GPUs. In the literature there are several "best practices" to set
    learning rates and schedules for large batch sizes.
    """

    if device_config.assigned_rank > 0:
        mute_current_process()

    logger.info("Distributed training starting...")
    if not torch.distributed.is_initialized():
        backend = "gloo" if os.name == "nt" else "nccl"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(device_config.assigned_rank)

    if torch.distributed.get_rank() == 0:
        logger.info(f"Training in distributed mode... with {str(torch.distributed.get_world_size())} GPUs")
    device_config.device = "cuda:%d" % device_config.assigned_rank


@record
def restart_script_with_ddp(num_gpus: int = None):
    """Launch the same script as the one that was launched (i.e. the command used to start the current process is re-used) but on subprocesses (i.e. with DDP).

    :param num_gpus: How many gpu's you want to run the script on. If not specified, every available device will be used.
    """
    ddp_port = find_free_port()

    # Get the value fom recipe if specified, otherwise take all available devices.
    num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
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
        run_id="sg_initiated",
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

    elastic_launch(config=config, entrypoint=sys.executable)(*sys.argv, *EXTRA_ARGS)

    # The code below should actually never be reached as the process will be in a loop inside elastic_launch until any subprocess crashes.
    sys.exit(0)


def get_gpu_mem_utilization():
    """GPU memory managed by the caching allocator in bytes for a given device."""

    # Workaround to work on any torch version
    if hasattr(torch.cuda, "memory_reserved"):
        return torch.cuda.memory_reserved()
    else:
        return torch.cuda.memory_cached()


class DDPNotSetupException(Exception):
    """Exception raised when DDP setup is required but was not done"""

    def __init__(self):
        self.message = (
            "Your environment was not setup correctly for DDP.\n"
            "Please run at the beginning of your script:\n"
            ">>> from super_gradients.training.utils.distributed_training_utils import setup_device'\n"
            ">>> from super_gradients.common.data_types.enum import MultiGPUMode\n"
            ">>> setup_device(multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=...)"
        )
        super().__init__(self.message)
