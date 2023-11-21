import logging
import os
import platform

import psutil

from super_gradients.common.environment.env_variables import env_variables


def mute_subprocesses():
    """Mute (prints, warnings and all logs except ERRORS) of some subprocesses to avoid having duplicates in the logs."""

    # When running DDP, mute all nodes except for the master node
    if int(env_variables.LOCAL_RANK) > 0:
        mute_current_process()

    mute_non_linux_dataloader_worker_process()


def mute_current_process():
    """Mute prints, warnings and all logs except ERRORS. This is meant when running multiple processes."""
    # Ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # Ignore prints
    import sys

    sys.stdout = open(os.devnull, "w", encoding="utf-8")

    # Only show ERRORS
    process_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in process_loggers:
        logger.setLevel(logging.ERROR)


def mute_non_linux_dataloader_worker_process() -> None:
    """Mute any worker process when running on mac/windows.
    This is required because the dataloader workers are "spawned" on mac/windows and "forked" on linux.
    The consequence being that the on mac/windows every module will be imported on each worker process, leading to a huge number of prints/logs that are
    displayed on import.
    For more information: https://pytorch.org/docs/stable/data.html#platform-specific-behaviors

    To avoid this, we mute the dataloader workers when running on mac/windows.

    Note:
        We assume that the process tree looks like this:
            Without DDP:
                ... -> main_process -> worker_process
            With DDP:
                ... -> main_process -> node_process -> worker_process

        Knowing that depending on how the script is launched, main_process might be child of other non "python" processes such as:
                ssh(non-python) -> pycharm(non-python) -> main_process(python) -> ...
    """

    if is_non_linux_dataloader_worker_process():
        mute_current_process()


def is_non_linux_dataloader_worker_process() -> bool:
    """Check if current process is a dataloader worker process on a non linux device."""
    if any(os_name in platform.platform() for os_name in ["macOS", "Windows"]):

        # When using DDP with SG launcher, we expect the worker process to have 2 parents processes using python, and only 1 otherwise.
        # Note that this is a "root_process" is the root process only if current process is a worker process
        if int(env_variables.LOCAL_RANK) == -1:
            # NO DDP
            main_process = psutil.Process().parent()
        elif os.environ.get("TORCHELASTIC_RUN_ID") == "sg_initiated":
            # DDP launched using SG logic
            main_process = psutil.Process().parent().parent()
        else:
            # DDP launched using torch.distributed.launch or torchrun
            main_process = psutil.Process().parent()

        is_worker_process = main_process and "python" in main_process.name()

        if is_worker_process:
            return True
    return False
