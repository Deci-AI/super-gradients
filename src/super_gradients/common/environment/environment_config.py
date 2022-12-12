import dataclasses
import os
import argparse

__all__ = ["PKG_CHECKPOINTS_DIR", "device_config"]

try:
    pass
except Exception:
    os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
    PKG_CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints")


# TODO: check this
def _get_assigned_rank():
    if os.getenv("LOCAL_RANK") is not None:
        return int(os.getenv("LOCAL_RANK"))
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", default=-1)
        args, _ = parser.parse_known_args()
        return args.local_rank


_DDP_ASSIGNED_RANK = _get_assigned_rank()


@dataclasses.dataclass
class DeviceConfig:
    device: str = None
    multi_gpu: str = None
    assigned_rank: str = dataclasses.field(default=_DDP_ASSIGNED_RANK, init=False)


device_config = DeviceConfig()
