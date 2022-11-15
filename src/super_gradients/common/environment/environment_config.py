import os

import pkg_resources

try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    PKG_CHECKPOINTS_DIR = os.path.join(os.path.expanduser("~/sg/"), "checkpoints")
    os.makedirs(PKG_CHECKPOINTS_DIR, exist_ok=True)

DDP_LOCAL_RANK = int(os.getenv("LOCAL_RANK", default=-1))
EXTRA_ARGS = []

INIT_TRAINER = False
