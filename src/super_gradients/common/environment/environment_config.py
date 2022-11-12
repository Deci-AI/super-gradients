import os

import pkg_resources

try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    PKG_CHECKPOINTS_DIR = os.path.join(os.path.expanduser("~/sg_logs/"), "checkpoints")
    os.makedirs(PKG_CHECKPOINTS_DIR, exist_ok=True)
    # os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True) # FIXME this creates in the working dir which is different when running
    #                                                                      # train_from_config and evaluate_from_config


DDP_LOCAL_RANK = -1

INIT_TRAINER = False
