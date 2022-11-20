from enum import Enum
import stringcase


class MultiGPUMode(str, Enum):
    """
    MultiGPUMode

        Attributes:
            OFF                       - Single GPU Mode / CPU Mode
            DATA_PARALLEL             - Multiple GPUs, Synchronous
            DISTRIBUTED_DATA_PARALLEL - Multiple GPUs, Asynchronous
    """

    OFF = "Off"
    DATA_PARALLEL = "DP"
    DISTRIBUTED_DATA_PARALLEL = "DDP"
    AUTO = "AUTO"

    @classmethod
    def dict(cls):
        """
        return dictionary mapping from the mode name (in call string cases) to the enum value
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
