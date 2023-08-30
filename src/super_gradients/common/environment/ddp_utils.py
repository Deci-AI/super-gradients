import socket
from functools import wraps
import os
from typing import Any, List, Callable

import torch
import torch.distributed as dist

from super_gradients.common.environment.device_utils import device_config
from super_gradients.common.environment.omegaconf_utils import register_hydra_resolvers
from super_gradients.common.environment.argparse_utils import pop_local_rank


def init_trainer():
    """
    Initialize the super_gradients environment.

    This function should be the first thing to be called by any code running super_gradients.
    """
    register_hydra_resolvers()
    pop_local_rank()


def is_distributed() -> bool:
    """Check if current process is a DDP subprocess."""
    return device_config.assigned_rank >= 0


def is_launched_using_sg():
    """Check if the current process is a subprocess launched using SG restart_script_with_ddp"""
    return os.environ.get("TORCHELASTIC_RUN_ID") == "sg_initiated"


def is_main_process():
    """Check if current process is considered as the main process (i.e. is responsible for sanity check, atexit upload, ...).
    The definition ensures that 1 and only 1 process follows this condition, regardless of how the run was started.

    The rule is as follow:
        - If not DDP: main process is current process
        - If DDP launched using SuperGradients: main process is the launching process (rank=-1)
        - If DDP launched with torch: main process is rank 0
    """

    if not is_distributed():  # If no DDP, or DDP launching process
        return True
    elif (
        device_config.assigned_rank == 0 and not is_launched_using_sg()
    ):  # If DDP launched using torch.distributed.launch or torchrun, we need to run the check on rank 0
        return True
    else:
        return False


def multi_process_safe(func):
    """
    A decorator for making sure a function runs only in main process.
    If not in DDP mode (local_rank = -1), the function will run.
    If in DDP mode, the function will run only in the main process (local_rank = 0)
    This works only for functions with no return value
    """

    def do_nothing(*args, **kwargs):
        pass

    @wraps(func)
    def wrapper(*args, **kwargs):
        if device_config.assigned_rank <= 0:
            return func(*args, **kwargs)
        else:
            return do_nothing(*args, **kwargs)

    return wrapper


def find_free_port() -> int:
    """Find an available port of current machine/node.
    Note: there is still a chance the port could be taken by other processes."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Binding to port 0 will cause the OS to find an available port for us
        sock.bind(("", 0))
        _ip, port = sock.getsockname()
    return port


def get_local_rank():
    """
    Returns the local rank if running in DDP, and 0 otherwise
    :return: local rank
    """
    return dist.get_rank() if dist.is_initialized() else 0


def require_ddp_setup() -> bool:
    from super_gradients.common import MultiGPUMode

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


def execute_and_distribute_from_master(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to execute a function on the master process and distribute the result to all other processes.
    Useful in parallel computing scenarios where a computational task needs to be performed only on the master
    node (e.g., a computational-heavy calculation), and the result must be shared with other nodes without
    redundant computation.

    Example usage:
        >>> @execute_and_distribute_from_master
        >>> def some_code_to_run(param1, param2):
        >>>     return param1 + param2

    The wrapped function will only be executed on the master node, and the result will be propagated to all
    other nodes.

    :param func:    The function to be executed on the master process and whose result is to be distributed.
    :return:        A wrapper function that encapsulates the execute-and-distribute logic.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Run the function only if it's the master process
        if device_config.assigned_rank <= 0:
            result = func(*args, **kwargs)
        else:
            result = None

        # Broadcast the result from the master process to all nodes
        return broadcast_from_master(result)

    return wrapper


def broadcast_from_master(data: Any) -> Any:
    """
    Broadcast data from master node to all other nodes. This may be required when you
    want to compute something only on master node (e.g computational-heavy metric) and
    don't want to waste CPU of other nodes doing the same work simultaneously.

    :param data:    Data to be broadcasted from master node (rank 0)
    :return:        Data from rank 0 node
    """
    world_size = get_world_size()
    if world_size == 1:
        return data
    broadcast_list = [data] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(broadcast_list, src=0)
    return broadcast_list[0]
