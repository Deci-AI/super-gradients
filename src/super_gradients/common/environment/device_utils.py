import os
import dataclasses

from super_gradients.common.environment.argparse_utils import pop_arg
from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__)


__all__ = ["device_config"]


def _get_assigned_rank() -> int:
    """Get the rank assigned by DDP launcher. If not DDP subprocess, return -1."""
    if os.getenv("LOCAL_RANK") is not None:
        return int(os.getenv("LOCAL_RANK"))
    else:
        local_rank = pop_arg("local_rank", default_value=-1)
        if local_rank != -1:
            logger.info("local_rank was automatically parsed from your config.")
        return local_rank


@dataclasses.dataclass
class DeviceConfig:
    device: str = None
    multi_gpu: str = None
    assigned_rank: str = dataclasses.field(default=_get_assigned_rank(), init=False)


# Singleton holding the device information
device_config = DeviceConfig()
