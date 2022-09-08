import sys
import random
import os
import hydra
import pkg_resources
import torch
from typing import Tuple, List
from argparse import ArgumentParser, Namespace

from omegaconf import DictConfig
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from super_gradients import Trainer
from super_gradients.common.environment import environment_config
from super_gradients.common.environment.env_helpers import setup_rank, is_distributed, init_trainer
from super_gradients.training import utils as core_utils
from super_gradients.common.data_types.enum import MultiGPUMode


# def parse_args() -> Tuple[Namespace, List[str]]:
#     """Get a parser that only parses args that are not (and should not be) included in the recipes."""
#     parser = ArgumentParser(description="Torch Distributed Elastic Training Launcher")
#     # parser.add_argument(
#     #     "--subprocess",
#     #     action='store_true',
#     #     help="When used, this flag indicates that the current process is a subprocess (DDP)",
#     # )
#     # parser.add_argument("--local_rank", type=int, default=-1)
#     args, recipe_args = parser.parse_known_args()
#     return args, recipe_args


# def setup() -> None:
#     """This step is required to differentiate subprocesses to main processes."""
#     # args, _ = parse_args()
#     # if args.subprocess:
#     #     # This is required because hydra does not allow this type of parameters
#     #     sys.argv.remove("--subprocess")
#     #     environment_config.IS_SUBPROCESS = True
#     setup_rank()


def _find_free_port() -> int:
    """Find an available port of current machine / node."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch_ddp(cfg: DictConfig) -> None:
    """Create a configuration to launch DDP on single node without restart.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    :return:    Configuration to start DDP"""

    # Get the value fom recipe if specified, otherwise take all available devices.
    nproc_per_node = core_utils.get_param(cfg, 'nproc_per_node', torch.cuda.device_count())
    ddp_port = _find_free_port()

    config = LaunchConfig(
        nproc_per_node=nproc_per_node,
        min_nodes=1,
        max_nodes=1,
        run_id='none',
        role='default',
        rdzv_endpoint=f'127.0.0.1:{ddp_port}',
        rdzv_backend='static',
        rdzv_configs={'rank': 0, 'timeout': 900},
        rdzv_timeout=-1,
        max_restarts=0,
        monitor_interval=5,
        start_method='spawn',
        log_dir=None,
        redirects=Std.NONE,
        tee=Std.NONE,
        metrics_cfg={})
    # _, recipe_args = parse_args()

    # elastic_launch(config=config, entrypoint=sys.executable)(sys.argv[0], *recipe_args)
    elastic_launch(config=config, entrypoint=sys.executable)(*sys.argv)


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""))
def run(cfg: DictConfig) -> None:
    """Launch the training job according to the specified recipe.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    multi_gpu = core_utils.get_param(cfg, 'multi_gpu', MultiGPUMode.OFF)

    if multi_gpu == MultiGPUMode.OFF or is_distributed():
        Trainer.train_from_config(cfg)
    else:
        launch_ddp(cfg)


@record
def main() -> None:
    """This script is designed to run a recipe with and without DDP.

    Can be launched in either way:
        - python -m run --config-name=<MY-RECIPE> [nproc_per_node=<NUM>]
        - python -m torch.distributed.launcher --nproc_per_node=<NUM> run.py --config-name=<MY-RECIPE>
    Note: When using the first approach, nproc_per_node will by default use the value specified in recipe.
    """
    init_trainer()
    run()


if __name__ == "__main__":
    main()
