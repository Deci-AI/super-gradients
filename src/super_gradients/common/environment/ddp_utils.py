import socket
from functools import wraps
import os
import pickle
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
    don't want to vaste CPU of other nodes doing same work simultaneously.

    >>> if device_config.assigned_rank <= 0:
    >>>    result = some_code_to_run(...)
    >>> else:
    >>>    result = None
    >>> # 'result' propagated to all nodes from master
    >>> result = broadcast_from_master(result)

    :param data:    Data to be broadcasted from master node (rank 0)
    :return:        Data from rank 0 node
    """
    world_size = get_world_size()
    if world_size == 1:
        return data

    local_rank = get_local_rank()
    storage: torch.Tensor

    if local_rank == 0:
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        payload = torch.ByteTensor(storage).to("cuda")
        local_size = payload.numel()
    else:
        local_size = 0

    # Propagate target tensor size to all nodes
    local_size = max(all_gather(local_size))
    if local_rank != 0:
        payload = torch.empty((local_size,), dtype=torch.uint8, device="cuda")

    dist.broadcast(payload, 0)
    buffer = payload.cpu().numpy().tobytes()
    return pickle.loads(buffer)


def all_gather(data: Any) -> List[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    :param data:    Any picklable object
    :return:        List of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    try:
        storage = torch.UntypedStorage.from_buffer(buffer, dtype=torch.uint8)
    except AttributeError:
        storage = torch._UntypedStorage.from_buffer(buffer, dtype=torch.uint8)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
