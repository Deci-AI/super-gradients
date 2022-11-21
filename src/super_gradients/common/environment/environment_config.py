import os

import pkg_resources

try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
    PKG_CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints")


DDP_LOCAL_RANK = int(os.getenv("LOCAL_RANK", default=-1))
EXTRA_ARGS = []

INIT_TRAINER = False
