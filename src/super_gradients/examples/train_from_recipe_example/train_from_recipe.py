"""
Example code for running SuperGradient's recipes with and without DDP.

Can be launched in either way:
    - python -m run --config-name=<MY-RECIPE> [nproc_per_node=<NUM>]
    - python -m torch.distributed.launcher --nproc_per_node=<NUM> train_from_recipe.py --config-name=<MY-RECIPE>
Note: When using the first approach, nproc_per_node will by default use the value specified in recipe.

For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory (src/super_gradients/recipes/).
"""

import sys
import hydra
import pkg_resources
import torch

from omegaconf import DictConfig
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from super_gradients import Trainer
from super_gradients.common.environment.env_helpers import is_distributed, init_trainer, find_free_port
from super_gradients.training import utils as core_utils
from super_gradients.common.data_types.enum import MultiGPUMode


def launch_ddp(cfg: DictConfig):
    """Create a configuration to launch DDP on single node without restart.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    :return:    Configuration to start DDP"""

    # Get the value fom recipe if specified, otherwise take all available devices.
    nproc_per_node = core_utils.get_param(cfg, 'nproc_per_node', torch.cuda.device_count())
    ddp_port = find_free_port()

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

    elastic_launch(config=config, entrypoint=sys.executable)(*sys.argv)


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def run(cfg: DictConfig):
    """Launch the training job according to the specified recipe.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    multi_gpu = core_utils.get_param(cfg, 'multi_gpu', MultiGPUMode.OFF)

    if multi_gpu == MultiGPUMode.OFF or is_distributed():
        Trainer.train_from_config(cfg)
    else:
        launch_ddp(cfg)


@record
def main():
    init_trainer()
    run()


if __name__ == "__main__":
    main()
