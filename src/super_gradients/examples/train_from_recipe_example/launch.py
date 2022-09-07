
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
logger = logging.getLogger() #TODO: replace with SG


DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(num_gpus_per_machine: int = 2):
    """
    """

    mp.spawn(
        _distributed_worker,
        nprocs=num_gpus_per_machine,
        args=(_find_free_port(),),
        # daemon=False,
        # start_method="spawn",
    )


def _distributed_worker(local_rank, ddp_port):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    assert local_rank <= torch.cuda.device_count()
    import os
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(8)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(ddp_port)

    logger.info("Rank {} initialization finished.".format(local_rank))
    FAULT_TIMEOUT = timedelta(minutes=1)
    dist.init_process_group(
        backend="nccl",
        # init_method=None,
        world_size=2,
        rank=local_rank,
        timeout=FAULT_TIMEOUT
    )

    def synchronize():
        """
        Helper function to synchronize (barrier) among all processes when using distributed training
        """
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        dist.barrier()

    synchronize()
    torch.cuda.set_device(local_rank)
    train()

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



@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def train(cfg: DictConfig):
    """Launch the training job according to the specified recipe.

    :param cfg: Hydra config that was specified when launching the job with --config-name
    """
    Trainer.train_from_config(cfg)


def setup_train():
    return train()

if __name__ == '__main__':
    launch()
