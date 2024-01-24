from typing import Optional
import dataclasses

import torch

from super_gradients.common.environment.env_variables import env_variables
from super_gradients.common.environment.argparse_utils import pop_local_rank


__all__ = ["device_config"]


def _get_assigned_rank() -> int:
    """Get the rank assigned by DDP launcher. If not DDP subprocess, return -1."""
    if env_variables.LOCAL_RANK != -1:
        return env_variables.LOCAL_RANK
    else:
        return pop_local_rank()


@dataclasses.dataclass
class DeviceConfig:
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpu: Optional[str] = None
    num_gpus: Optional[str] = None
    assigned_rank: int = dataclasses.field(default=_get_assigned_rank(), init=False)

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        if "cuda" in value and not torch.cuda.is_available():
            raise ValueError("CUDA is not available, cannot set device to cuda")
        self._device = value

    @property
    def is_cuda(self):
        return "cuda" in self._device


# Singleton holding the device information
device_config = DeviceConfig()
