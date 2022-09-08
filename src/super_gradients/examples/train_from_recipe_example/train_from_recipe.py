import os
import random
import hydra
import pkg_resources
import torch
from typing import Tuple
from omegaconf import DictConfig

from super_gradients import Trainer
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.common.environment.env_helpers import launch_sub_processes, init_local_process, kill_subprocesses, register_hydra_resolvers, init_trainer
from super_gradients.training import utils as core_utils

#
# @hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
# def main(cfg: DictConfig):
#     """Launch a training on single node or multiple node depending on the specified config.
#
#     :param cfg: Hydra config that was specified when launching the job with --config-name
#     """
#     multi_gpu = core_utils.get_param(cfg, 'multi_gpu')
#
#     if not multi_gpu or multi_gpu == MultiGPUMode.OFF:
#         train(cfg)
#     else:
#         local_rank, ddp_port, nproc_per_node = get_ddp_params(cfg)
#         subprocesses = launch_sub_processes(ddp_port, world_size=nproc_per_node) if local_rank == 0 else []
#         init_local_process(local_rank, ddp_port, world_size=nproc_per_node)
#
#         try:
#             train(cfg)
#         finally:
#             kill_subprocesses(subprocesses)


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def train_from_recipe(cfg):
    """Launch the training job according to the specified recipe.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    Trainer.train_from_config(cfg)


def main():
    train_from_recipe()


if __name__ == "__main__":
    os.environ["RANK"] = os.getenv("LOCAL_RANK", "0")
    init_trainer()
    main()
