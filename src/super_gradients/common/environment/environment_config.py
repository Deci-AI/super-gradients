import os
import sys

import pkg_resources

try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
    PKG_CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints")


DDP_LOCAL_RANK = int(os.getenv("LOCAL_RANK", default=-1))
EXTRA_ARGS = []

# This needs to be run at the beginning because later on we drop --local_rank
IS_TORCH_DISTRIBUTED_LAUNCH = any("--local_rank" in arg for arg in sys.argv)

INIT_TRAINER = False
