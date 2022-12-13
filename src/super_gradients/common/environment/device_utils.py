import os
import dataclasses

import torch

from super_gradients.common.environment.argparse_utils import pop_local_rank


__all__ = ["device_config"]


def _get_assigned_rank() -> int:
    """Get the rank assigned by DDP launcher. If not DDP subprocess, return -1."""
    if os.getenv("LOCAL_RANK") is not None:
        return int(os.getenv("LOCAL_RANK"))
    else:
        return pop_local_rank()


@dataclasses.dataclass
class DeviceConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpu: str = None
    assigned_rank: str = dataclasses.field(default=_get_assigned_rank(), init=False)


# Singleton holding the device information
device_config = DeviceConfig()
