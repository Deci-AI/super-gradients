from enum import Enum


class MultiGPUMode(str, Enum):
    """
    MultiGPUMode

        Attributes:
            OFF                       - Single GPU Mode / CPU Mode
            DATA_PARALLEL             - Multiple GPUs, Synchronous
            DISTRIBUTED_DATA_PARALLEL - Multiple GPUs, Asynchronous
    """
    OFF = 'Off'
    DATA_PARALLEL = 'DP'
    DISTRIBUTED_DATA_PARALLEL = 'DDP'
    AUTO = "AUTO"
