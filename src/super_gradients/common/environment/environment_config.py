import dataclasses
import os
import argparse
import pkg_resources

__all__ = ["PKG_CHECKPOINTS_DIR", "EXTRA_ARGS", "device_config"]


try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
    PKG_CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints")


def _get_requested_rank():
    if os.getenv("LOCAL_RANK") is not None:
        return int(os.getenv("LOCAL_RANK"))
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", default=-1)
        args, _ = parser.parse_known_args()
        return args.local_rank


_DDP_REQUESTED_RANK = _get_requested_rank()

EXTRA_ARGS = []


@dataclasses.dataclass
class DeviceConfig:
    device: str = None
    multi_gpu: str = None
    requested_rank: str = dataclasses.field(default=_DDP_REQUESTED_RANK, init=False)


device_config = DeviceConfig()
