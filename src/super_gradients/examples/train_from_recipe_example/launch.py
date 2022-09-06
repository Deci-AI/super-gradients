# #
# # #!/usr/bin/env python3
# # # -*- coding:utf-8 -*-
# # # Code are based on
# # # https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# # # Copyright (c) Facebook, Inc. and its affiliates.
# # # Copyright (c) Megvii, Inc. and its affiliates.
# #
# # import sys
# # from datetime import timedelta
# #
# # import torch
# # import torch.distributed as dist
# # import torch.multiprocessing as mp
# # import logging
# # logger = logging.getLogger() #TODO: replace with SG
# #
# #
# # DEFAULT_TIMEOUT = timedelta(minutes=30)
# #
# #
# # def _find_free_port():
# #     """
# #     Find an available port of current machine / node.
# #     """
# #     import socket
# #
# #     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# #     # Binding to port 0 will cause the OS to find an available port for us
# #     sock.bind(("", 0))
# #     port = sock.getsockname()[1]
# #     sock.close()
# #     # NOTE: there is still a chance the port could be taken by other processes.
# #     return port
# #
# #
# # def launch(num_gpus_per_machine: int = 3):
# #     """
# #     """
# #
# #     mp.spawn(
# #         _distributed_worker,
# #         nprocs=num_gpus_per_machine,
# #         args=(_find_free_port(),),
# #         # daemon=False,
# #         # start_method="spawn",
# #     )
# #
# #
# # def _distributed_worker(local_rank, ddp_port):
# #     assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
# #     assert local_rank <= torch.cuda.device_count()
# #     import os
# #     os.environ['RANK'] = str(local_rank)
# #     os.environ['WORLD_SIZE'] = str(8)
# #     os.environ['MASTER_ADDR'] = '127.0.0.1'
# #     os.environ['MASTER_PORT'] = str(ddp_port)
# #
# #     logger.info("Rank {} initialization finished.".format(local_rank))
# #     FAULT_TIMEOUT = timedelta(minutes=1)
# #     dist.init_process_group(
# #         backend="c10d",
# #         # init_method=None,
# #         world_size=8,
# #         rank=local_rank,
# #         timeout=FAULT_TIMEOUT
# #     )
# #
# #     def synchronize():
# #         """
# #         Helper function to synchronize (barrier) among all processes when using distributed training
# #         """
# #         if not dist.is_available():
# #             return
# #         if not dist.is_initialized():
# #             return
# #         world_size = dist.get_world_size()
# #         if world_size == 1:
# #             return
# #         dist.barrier()
# #
# #     synchronize()
# #     torch.cuda.set_device(local_rank)
# #     train()
# #
# # import os
# # import random
# # import hydra
# # import pkg_resources
# # import torch
# # from typing import Tuple
# # from omegaconf import DictConfig
# #
# # from super_gradients import Trainer
# # from super_gradients.common.data_types.enum import MultiGPUMode
# # from super_gradients.common.environment.env_helpers import launch_sub_processes, init_local_process, kill_subprocesses, register_hydra_resolvers, init_trainer
# # from super_gradients.training import utils as core_utils
# #
# #
# #
# # @hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
# # def train(cfg: DictConfig):
# #     """Launch the training job according to the specified recipe.
# #
# #     :param cfg: Hydra config that was specified when launching the job with --config-name
# #     """
# #     Trainer.train_from_config(cfg)
# #
# #
# # if __name__ == '__main__':
# #     launch()
#
#
# from torch.distributed.launcher.api import LaunchConfig, elastic_launch
# import os
#
# def my_process(args):
#     env_variables = [
#         "RANK",
#         "WORLD_SIZE",
#         "MASTER_ADDR",
#         "MASTER_PORT",
#         "TORCHELASTIC_MAX_RESTARTS",
#         # etc...
#     ]
#     for env_var in env_variables:
#         print(f"{env_var}: {os.environ.get(env_var)}")
#     import time
#     time.sleep(30)
#     print(f"here are my own args: {args}")
#
# if __name__ == "__main__":
#     config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=3, rdzv_endpoint="localhost:0", rdzv_backend="static")
#     # dist.init_process_group(backend="nccl")
#     outputs = elastic_launch(config, my_process)("my args")


#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
``torchrun`` provides a superset of the functionality as ``torch.distributed.launch``
with the following additional functionalities:

1. Worker failures are handled gracefully by restarting all workers.

2. Worker ``RANK`` and ``WORLD_SIZE`` are assigned automatically.

3. Number of nodes is allowed to change between minimum and maximum sizes (elasticity).

.. note:: ``torchrun`` is a python
          `console script <https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts>`_
          to the main module
          `torch.distributed.run <https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py>`_
          declared in the ``entry_points`` configuration in
          `setup.py <https://github.com/pytorch/pytorch/blob/master/setup.py>`_.
          It is equivalent to invoking ``python -m torch.distributed.run``.


Transitioning from torch.distributed.launch to torchrun
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


``torchrun`` supports the same arguments as ``torch.distributed.launch`` **except**
for ``--use_env`` which is now deprecated. To migrate from ``torch.distributed.launch``
to ``torchrun`` follow these steps:

1.  If your training script is already reading ``local_rank`` from the ``LOCAL_RANK`` environment variable.
    Then you need simply omit the ``--use_env`` flag, e.g.:

    +--------------------------------------------------------------------+--------------------------------------------+
    |         ``torch.distributed.launch``                               |                ``torchrun``                |
    +====================================================================+============================================+
    |                                                                    |                                            |
    | .. code-block:: shell-session                                      | .. code-block:: shell-session              |
    |                                                                    |                                            |
    |    $ python -m torch.distributed.launch --use_env train_script.py  |    $ torchrun train_script.py              |
    |                                                                    |                                            |
    +--------------------------------------------------------------------+--------------------------------------------+

2.  If your training script reads local rank from a ``--local_rank`` cmd argument.
    Change your training script to read from the ``LOCAL_RANK`` environment variable as
    demonstrated by the following code snippet:

    +-------------------------------------------------------+----------------------------------------------------+
    |         ``torch.distributed.launch``                  |                    ``torchrun``                    |
    +=======================================================+====================================================+
    |                                                       |                                                    |
    | .. code-block:: python                                | .. code-block:: python                             |
    |                                                       |                                                    |
    |                                                       |                                                    |
    |    import argparse                                    |     import os                                      |
    |    parser = argparse.ArgumentParser()                 |     local_rank = int(os.environ["LOCAL_RANK"])     |
    |    parser.add_argument("--local_rank", type=int)      |                                                    |
    |    args = parser.parse_args()                         |                                                    |
    |                                                       |                                                    |
    |    local_rank = args.local_rank                       |                                                    |
    |                                                       |                                                    |
    +-------------------------------------------------------+----------------------------------------------------+

The aformentioned changes suffice to migrate from ``torch.distributed.launch`` to ``torchrun``.
To take advantage of new features such as elasticity, fault-tolerance, and error reporting of ``torchrun``
please refer to:

* :ref:`elastic_train_script` for more information on authoring training scripts that are ``torchrun`` compliant.
* the rest of this page for more information on the features of ``torchrun``.


Usage
--------

Single-node multi-worker
++++++++++++++++++++++++++++++

::

    torchrun
        --standalone
        --nnodes=1
        --nproc_per_node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

Stacked single-node multi-worker
+++++++++++++++++++++++++++++++++++

To run multiple instances (separate jobs) of single-node, multi-worker on the
same host, we need to make sure that each instance (job) is
setup on different ports to avoid port conflicts (or worse, two jobs being merged
as a single job). To do this you have to run with ``--rdzv_backend=c10d``
and specify a different port by setting ``--rdzv_endpoint=localhost:$PORT_k``.
For ``--nodes=1``, its often convenient to let ``torchrun`` pick a free random
port automatically instead of manually assgining different ports for each run.

::

    torchrun
        --rdzv_backend=c10d
        --rdzv_endpoint=localhost:0
        --nnodes=1
        --nproc_per_node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


Fault tolerant (fixed sized number of workers, no elasticity, tolerates 3 failures)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=$NUM_NODES
        --nproc_per_node=$NUM_TRAINERS
        --max_restarts=3
        --rdzv_id=$JOB_ID
        --rdzv_backend=c10d
        --rdzv_endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

Elastic (``min=1``, ``max=4``, tolerates up to 3 membership changes or failures)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=1:4
        --nproc_per_node=$NUM_TRAINERS
        --max_restarts=3
        --rdzv_id=$JOB_ID
        --rdzv_backend=c10d
        --rdzv_endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

Note on rendezvous backend
------------------------------

For multi-node training you need to specify:

1. ``--rdzv_id``: A unique job id (shared by all nodes participating in the job)
2. ``--rdzv_backend``: An implementation of
   :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler`
3. ``--rdzv_endpoint``: The endpoint where the rendezvous backend is running; usually in form
   ``host:port``.

Currently ``c10d`` (recommended), ``etcd-v2``, and ``etcd`` (legacy)  rendezvous backends are
supported out of the box. To use ``etcd-v2`` or ``etcd``, setup an etcd server with the ``v2`` api
enabled (e.g. ``--enable-v2``).

.. warning::
   ``etcd-v2`` and ``etcd`` rendezvous use etcd API v2. You MUST enable the v2 API on the etcd
   server. Our tests use etcd v3.4.3.

.. warning::
   For etcd-based rendezvous we recommend using ``etcd-v2`` over ``etcd`` which is functionally
   equivalent, but uses a revised implementation. ``etcd`` is in maintenance mode and will be
   removed in a future version.

Definitions
--------------

1. ``Node`` - A physical instance or a container; maps to the unit that the job manager works with.

2. ``Worker`` - A worker in the context of distributed training.

3. ``WorkerGroup`` - The set of workers that execute the same function (e.g. trainers).

4. ``LocalWorkerGroup`` - A subset of the workers in the worker group running on the same node.

5. ``RANK`` - The rank of the worker within a worker group.

6. ``WORLD_SIZE`` - The total number of workers in a worker group.

7. ``LOCAL_RANK`` - The rank of the worker within a local worker group.

8. ``LOCAL_WORLD_SIZE`` - The size of the local worker group.

9. ``rdzv_id`` - A user-defined id that uniquely identifies the worker group for a job. This id is
   used by each node to join as a member of a particular worker group.

9. ``rdzv_backend`` - The backend of the rendezvous (e.g. ``c10d``). This is typically a strongly
   consistent key-value store.

10. ``rdzv_endpoint`` - The rendezvous backend endpoint; usually in form ``<host>:<port>``.

A ``Node`` runs ``LOCAL_WORLD_SIZE`` workers which comprise a ``LocalWorkerGroup``. The union of
all ``LocalWorkerGroups`` in the nodes in the job comprise the ``WorkerGroup``.

Environment Variables
----------------------

The following environment variables are made available to you in your script:

1. ``LOCAL_RANK`` -  The local rank.

2. ``RANK`` -  The global rank.

3. ``GROUP_RANK`` - The rank of the worker group. A number between 0 and ``max_nnodes``. When
   running a single worker group per node, this is the rank of the node.

4. ``ROLE_RANK`` -  The rank of the worker across all the workers that have the same role. The role
   of the worker is specified in the ``WorkerSpec``.

5. ``LOCAL_WORLD_SIZE`` - The local world size (e.g. number of workers running locally); equals to
   ``--nproc_per_node`` specified on ``torchrun``.

6. ``WORLD_SIZE`` - The world size (total number of workers in the job).

7. ``ROLE_WORLD_SIZE`` - The total number of workers that was launched with the same role specified
   in ``WorkerSpec``.

8. ``MASTER_ADDR`` - The FQDN of the host that is running worker with rank 0; used to initialize
   the Torch Distributed backend.

9. ``MASTER_PORT`` - The port on the ``MASTER_ADDR`` that can be used to host the C10d TCP store.

10. ``TORCHELASTIC_RESTART_COUNT`` - The number of worker group restarts so far.

11. ``TORCHELASTIC_MAX_RESTARTS`` - The configured maximum number of restarts.

12. ``TORCHELASTIC_RUN_ID`` - Equal to the rendezvous ``run_id`` (e.g. unique job id).

13. ``PYTHON_EXEC`` - System executable override. If provided, the python user script will
    use the value of ``PYTHON_EXEC`` as executable. The `sys.executable` is used by default.

Deployment
------------

1. (Not needed for the C10d backend) Start the rendezvous backend server and get the endpoint (to be
   passed as ``--rdzv_endpoint`` to the launcher script)

2. Single-node multi-worker: Start the launcher on the host to start the agent process which
   creates and monitors a local worker group.

3. Multi-node multi-worker: Start the launcher with the same arguments on all the nodes
   participating in training.

When using a job/cluster manager the entry point command to the multi-node job should be this
launcher.

Failure Modes
---------------

1. Worker failure: For a training job with ``n`` workers, if ``k<=n`` workers fail all workers
   are stopped and restarted up to ``max_restarts``.

2. Agent failure: An agent failure results in a local worker group failure. It is up to the job
   manager to fail the entire job (gang semantics) or attempt to replace the node. Both behaviors
   are supported by the agent.

3. Node failure: Same as agent failure.

Membership Changes
--------------------

1. Node departure (scale-down): The agent is notified of the departure, all existing workers are
   stopped, a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

2. Node arrival (scale-up): The new node is admitted to the job, all existing workers are stopped,
   a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

Important Notices
--------------------

1. This utility and multi-process distributed (single-node or
   multi-node) GPU training currently only achieves the best performance using
   the NCCL distributed backend. Thus NCCL backend is the recommended backend to
   use for GPU training.

2. The environment variables necessary to initialize a Torch process group are provided to you by
   this module, no need for you to pass ``RANK`` manually.  To initialize a process group in your
   training script, simply run:

::

 >>> # xdoctest: +SKIP("stub")
 >>> import torch.distributed as dist
 >>> dist.init_process_group(backend="gloo|nccl")

3. In your training program, you can either use regular distributed functions
   or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
   training program uses GPUs for training and you would like to use
   :func:`torch.nn.parallel.DistributedDataParallel` module,
   here is how to configure it.

::

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[int(os.environ("LOCAL_RANK"))]``,
and ``output_device`` needs to be ``int(os.environ("LOCAL_RANK"))`` in order to use this
utility


4. On failures or membership changes ALL surviving workers are killed immediately. Make sure to
   checkpoint your progress. The frequency of checkpoints should depend on your job's tolerance
   for lost work.

5. This module only supports homogeneous ``LOCAL_WORLD_SIZE``. That is, it is assumed that all
   nodes run the same number of local workers (per role).

6. ``RANK`` is NOT stable. Between restarts, the local workers on a node can be assgined a
   different range of ranks than before. NEVER hard code any assumptions about the stable-ness of
   ranks or some correlation between ``RANK`` and ``LOCAL_RANK``.

7. When using elasticity (``min_size!=max_size``) DO NOT hard code assumptions about
   ``WORLD_SIZE`` as the world size can change as nodes are allowed to leave and join.

8. It is recommended for your script to have the following structure:

::

  def main():
    load_checkpoint(checkpoint_path)
    initialize()
    train()

  def train():
    for batch in iter(dataset):
      train_step(batch)

      if should_checkpoint:
        save_checkpoint(checkpoint_path)

9. (Recommended) On worker errors, this tool will summarize the details of the error
   (e.g. time, rank, host, pid, traceback, etc). On each node, the first error (by timestamp)
   is heuristically reported as the "Root Cause" error. To get tracebacks as part of this
   error summary print out, you must decorate your main entrypoint function in your
   training script as shown in the example below. If not decorated, then the summary
   will not include the traceback of the exception and will only contain the exitcode.
   For details on torchelastic error handling see: https://pytorch.org/docs/stable/elastic/errors.html

::

  from torch.distributed.elastic.multiprocessing.errors import record

  @record
  def main():
      # do train
      pass

  if __name__ == "__main__":
      main()

"""
import logging
import os
import sys
import uuid
from argparse import REMAINDER, ArgumentParser
from typing import Callable, List, Tuple, Union

import torch
from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import LaunchConfig, elastic_launch


log = get_logger()


def parse_args() -> ArgumentParser:
    """Helper function parsing the command line options."""

    parser = ArgumentParser(description="Torch Distributed Elastic Training Launcher")

    #
    # Worker/node size related arguments.
    #

    parser.add_argument(
        "--nnodes",
        action=env,
        type=str,
        default="1:1",
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    )
    parser.add_argument(
        "--nproc_per_node",
        action=env,
        type=str,
        default="1",
        help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
    )

    #
    # Rendezvous related arguments
    #

    parser.add_argument(
        "--rdzv_backend",
        action=env,
        type=str,
        default="static",
        help="Rendezvous backend.",
    )
    parser.add_argument(
        "--rdzv_endpoint",
        action=env,
        type=str,
        default="",
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv_id",
        action=env,
        type=str,
        default="none",
        help="User-defined group id.",
    )
    parser.add_argument(
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
        "on port 29400. Useful when launching single-node, multi-worker job. If specified "
        "--rdzv_backend, --rdzv_endpoint, --rdzv_id are auto-assigned; any explicitly set values "
        "are ignored.",
    )

    #
    # User-code launch related arguments.
    #

    parser.add_argument(
        "--max_restarts",
        action=env,
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--monitor_interval",
        action=env,
        type=float,
        default=5,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method to use when creating workers.",
    )
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    parser.add_argument(
        "-m",
        "--module",
        action=check_env,
        help="Change each process to interpret the launch script as a Python module, executing "
        "with the same behavior as 'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        action=check_env,
        help="Skip prepending the training script with 'python' - just execute it directly. Useful "
        "when the script is not a Python script.",
    )

    parser.add_argument(
        "--run_path",
        action=check_env,
        help="Run the training script with runpy.run_path in the same interpreter."
        " Script must be provided as an abs path (e.g. /abs/path/script.py)."
        " Takes precedence over --no_python.",
    )
    parser.add_argument(
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
        "directory is re-used for multiple runs (a unique job-level sub-directory is created with "
        "rdzv_id as the prefix).",
    )
    parser.add_argument(
        "-r",
        "--redirects",
        action=env,
        type=str,
        default="0",
        help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
        "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
        "stderr for local rank 1).",
    )
    parser.add_argument(
        "-t",
        "--tee",
        action=env,
        type=str,
        default="0",
        help="Tee std streams into a log file and also to console (see --redirects for format).",
    )

    #
    # Backwards compatible parameters with caffe2.distributed.launch.
    #

    parser.add_argument(
        "--node_rank",
        type=int,
        action=env,
        default=0,
        help="Rank of the node for multi-node distributed training.",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        action=env,
        help="Address of the master node (rank 0). It should be either the IP address or the "
        "hostname of rank 0. For single node multi-proc training the --master_addr can simply be "
        "127.0.0.1; IPv6 should have the pattern `[0:0:0:0:0:0:0:1]`.",
    )
    parser.add_argument(
        "--master_port",
        default=29500,
        type=int,
        action=env,
        help="Port on the master node (rank 0) to be used for communication during distributed "
        "training.",
    )

    #
    # Positional arguments.
    #

    # parser.add_argument(
    #     "training_script",
    #     type=str,
    #     help="Full path to the (single GPU) training program/script to be launched in parallel, "
    #     "followed by all the arguments for the training script.",
    # )
    #
    # # Rest from the training program.
    # parser.add_argument("training_script_args", nargs=REMAINDER)
    x, _ = parser.parse_known_args()
    return x


def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')

    return min_nodes, max_nodes


def determine_local_world_size(nproc_per_node: str):
    try:
        logging.info(f"Using nproc_per_node={nproc_per_node}.")
        return int(nproc_per_node)
    except ValueError:
        if nproc_per_node == "cpu":
            num_proc = os.cpu_count()
            device_type = "cpu"
        elif nproc_per_node == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available.")
            device_type = "gpu"
            num_proc = torch.cuda.device_count()
        elif nproc_per_node == "auto":
            if torch.cuda.is_available():
                num_proc = torch.cuda.device_count()
                device_type = "gpu"
            else:
                num_proc = os.cpu_count()
                device_type = "cpu"
        else:
            raise ValueError(f"Unsupported nproc_per_node value: {nproc_per_node}")

        log.info(
            f"Using nproc_per_node={nproc_per_node},"
            f" seting to {num_proc} since the instance "
            f"has {os.cpu_count()} {device_type}"
        )
        return num_proc


def get_rdzv_endpoint(args):
    if args.rdzv_backend == "static" and not args.rdzv_endpoint:
        return f"{args.master_addr}:{args.master_port}"
    return args.rdzv_endpoint


def get_use_env(args) -> bool:
    """
    Retrieves ``use_env`` from the args.
    ``use_env`` is a legacy argument, if ``use_env`` is False, the
    ``--node_rank`` argument will be transferred to all worker processes.
    ``use_env`` is only used by the ``torch.distributed.launch`` and will
    be deprecated in future releases.
    """
    if not hasattr(args, "use_env"):
        return True
    return args.use_env


def config_from_args(args) -> Tuple[LaunchConfig, Union[Callable, str], List[str]]:
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts >= 0

    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        log.warning(
            f"\n*****************************************\n"
            f"Setting OMP_NUM_THREADS environment variable for each process to be "
            f"{omp_num_threads} in default, to avoid your system being overloaded, "
            f"please further tune the variable for optimal performance in "
            f"your application as needed. \n"
            f"*****************************************"
        )
        # This env variable will be passed down to the subprocesses
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank

    rdzv_endpoint = get_rdzv_endpoint(args)

    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=3,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=rdzv_configs,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        redirects=Std.from_str(args.redirects),
        tee=Std.from_str(args.tee),
        log_dir=args.log_dir,
    )

    with_python = not args.no_python
    cmd: Union[Callable, str]
    cmd_args = []
    use_env = get_use_env(args)
    return config, "train_from_recipe.py", cmd_args


def run_script_path():
    """
    Runs the provided `training_script` from within this interpreter.
    Usage: `script_as_function("/abs/path/to/script.py", "--arg1", "val1")`
    """
    import runpy
    import sys

    from super_gradients.examples.train_from_recipe_example.train_from_recipe import train
    train()
    # sys.argv = [training_script] + [*training_script_args]
    # runpy.run_path(sys.argv[0], run_name="__main__")


def run(args, cfg):
    if args.standalone:
        args.rdzv_backend = "c10d"
        args.rdzv_endpoint = "localhost:29400"
        args.rdzv_id = str(uuid.uuid4())
        log.info(
            f"\n**************************************\n"
            f"Rendezvous info:\n"
            f"--rdzv_backend={args.rdzv_backend} "
            f"--rdzv_endpoint={args.rdzv_endpoint} "
            f"--rdzv_id={args.rdzv_id}\n"
            f"**************************************\n"
        )

    config, cmd, cmd_args = config_from_args(args)

    from super_gradients.examples.train_from_recipe_example.train_from_recipe import train, setup_train
    # from functools import partial

    elastic_launch(config=config, entrypoint=setup_train)()


import os
import hydra
import pkg_resources
import torch


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg):
    args = parse_args()
    run(args, cfg)


if __name__ == "__main__":
    main()