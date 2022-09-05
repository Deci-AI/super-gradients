import os
import random
import sys
import subprocess
import hydra
import pkg_resources
import torch
from typing import List, Tuple
from omegaconf import DictConfig

from super_gradients import Trainer, init_trainer
from super_gradients.common.environment import environment_config
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training import utils as core_utils


def train(cfg: DictConfig) -> None:
    """Launch the training job according to the specified recipe.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    Trainer.train_from_config(cfg)


def launch_sub_processes(ddp_port: int, n_subprocess: int) -> List[subprocess.Popen]:
    """Launch the job on all available nodes.

    :param ddp_port:        Port that will be used on every node
    :param n_subprocess:    Number of subprocesses to launch
    :return:                List of the subprocesses that were launched
    """
    subprocesses = []
    for i in range(n_subprocess):
        argv = sys.argv.copy() + [f'+local_rank={i}', f'+ddp_port={ddp_port}', ]
        subproc = subprocess.Popen([sys.executable, *argv], env=os.environ)
        subprocesses.append(subproc)
        print(f'Launched node {i} with pid={subproc.pid}')
    return subprocesses


def init_local_process(local_rank: int, ddp_port: int, world_size: int) -> None:
    """Initialize the local node with its rank and port.

    :param local_rank:  Local rank of this node
    :param ddp_port:    Port that will be used on this node
    :param world_size:  Number of nodes used for the training
    """

    environment_config.DDP_LOCAL_RANK = local_rank

    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(ddp_port)


def kill_subprocesses(subprocesses: List[subprocess.Popen]) -> None:
    """Kill all the subprocesses.

    :param subprocesses: All the subprocesses that were launched by this node
    """
    for process in subprocesses:
        print(f'Killing process pid={process.pid}')
        process.kill()


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""))
def main(cfg: DictConfig):
    """Launch a training on single node or multiple node depending on the specified config.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    init_trainer()
    multi_gpu = core_utils.get_param(cfg, 'multi_gpu')

    if not multi_gpu or multi_gpu == MultiGPUMode.OFF:
        train(cfg)
    else:
        local_rank, ddp_port, n_proc_node = get_ddp_params(cfg)
        subprocesses = launch_sub_processes(ddp_port, n_subprocess=n_proc_node-1) if local_rank == 0 else []
        init_local_process(local_rank, ddp_port, world_size=n_proc_node)

        try:
            train(cfg)
        finally:
            kill_subprocesses(subprocesses)


def get_ddp_params(cfg: DictConfig) -> Tuple[int, int, int]:
    """Get the DDP params. Take it from config file if set there, or set default value otherwise.

    :param cfg: Hydra config that was specified when launching the job with --config-name"""
    local_rank = core_utils.get_param(cfg, 'local_rank', 0 if torch.cuda.device_count() > 1 else -1)
    ddp_port = core_utils.get_param(cfg, 'ddp_port', random.randint(1025, 65536))
    n_proc_node = core_utils.get_param(cfg, 'n_proc_node', torch.cuda.device_count())
    return local_rank, ddp_port, n_proc_node


if __name__ == "__main__":
    main()
