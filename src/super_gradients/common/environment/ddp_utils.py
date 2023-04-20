import os
import socket
from functools import wraps


def init_trainer():
    """
    Initialize the super_gradients environment.

    This function should be the first thing to be called by any code running super_gradients.
    """
    from super_gradients.sanity_check import env_sanity_check
    from super_gradients.common.crash_handler import setup_crash_handler
    from super_gradients.common.environment.mute_processes import mute_subprocesses

    # register_hydra_resolvers()

    env_sanity_check()

    setup_crash_handler()
    # Mute on import to avoid the import prints/logs on sub processes
    mute_subprocesses()


def is_distributed() -> bool:
    """Check if current process is a DDP subprocess."""
    from super_gradients.common.environment.device_utils import device_config

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
    from super_gradients.common.environment.device_utils import device_config

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
        from super_gradients.common.environment.device_utils import device_config

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
