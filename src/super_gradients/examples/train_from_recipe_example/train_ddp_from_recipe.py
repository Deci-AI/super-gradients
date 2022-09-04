import os
import random
import sys
import subprocess
import hydra
import pkg_resources
import torch
from typing import List
from omegaconf import DictConfig

from super_gradients import Trainer
from super_gradients.common.environment import environment_config
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training import utils as core_utils


def train(cfg: DictConfig) -> None:
    """Launch the training job according to the specified recipe.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    Trainer.train_from_config(cfg)


def launch_sub_processes(ddp_port: int) -> List[subprocess.Popen]:
    """Launch the job on all available nodes.

    :param ddp_port:    Port that will be used on every node
    :return:            List of the subprocesses that were launched
    """
    subprocesses = []
    for i in range(1, torch.cuda.device_count()):
        argv = sys.argv.copy() + [f'+local_rank={i}', f'+ddp_port={ddp_port}']
        subproc = subprocess.Popen([sys.executable, *argv], env=os.environ)
        subprocesses.append(subproc)
        print(f'Launched node {i} with pid={subproc.pid}')
    return subprocesses


def init_local_process(local_rank: int, ddp_port: int) -> None:
    """Initialize the local node with its rank and port.

    :param local_rank:  Local rank of this node
    :param ddp_port:    Port that will be used on this node
    """

    environment_config.DDP_LOCAL_RANK = local_rank

    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
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
def main(cfg):
    """Launch a training on single node or multiple node depending on the specified config.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    multi_gpu = core_utils.get_param(cfg, 'multi_gpu')

    if not multi_gpu or multi_gpu == MultiGPUMode.OFF:
        train(cfg)
    else:
        local_rank = core_utils.get_param(cfg, 'local_rank', 0 if torch.cuda.device_count() > 1 else -1)
        ddp_port = core_utils.get_param(cfg, 'ddp_port', random.randint(1025, 65536))

        subprocesses = launch_sub_processes(ddp_port) if local_rank == 0 else []
        init_local_process(local_rank, ddp_port)

        try:
            train(cfg)
        finally:
            kill_subprocesses(subprocesses)


if __name__ == "__main__":
    main()
