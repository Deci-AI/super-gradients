from enum import Enum


class DownSampleMode(Enum):
    MAX_POOL = "max_pool"
    ANTI_ALIAS = "anti_alias"
