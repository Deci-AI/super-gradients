from typing import Dict
from enum import Enum
import stringcase


class MultiGPUMode(str, Enum):
    """MultiGPUMode: Enumeration of different ways to use gpu."""

    OFF = "Off"
    """Single GPU Mode / CPU Mode"""

    DATA_PARALLEL = "DP"
    """Multiple GPUs, Synchronous"""

    DISTRIBUTED_DATA_PARALLEL = "DDP"
    """Multiple GPUs, Asynchronous"""

    AUTO = "AUTO"
    """Runs "DDP" if more than 1 GPU available. Otherwise, runs "Off"."""

    @classmethod
    def dict(cls) -> Dict[str, "MultiGPUMode"]:
        """
        Return dictionary mapping from the mode name (in call string cases) to the enum value
        """
        out_dict = dict()
        for mode in MultiGPUMode:
            out_dict[mode.value] = mode
            out_dict[mode.name] = mode
            out_dict[stringcase.capitalcase(mode.name)] = mode
            out_dict[stringcase.camelcase(mode.name)] = mode
            out_dict[stringcase.lowercase(mode.name)] = mode
        out_dict[False] = MultiGPUMode.OFF
        return out_dict
