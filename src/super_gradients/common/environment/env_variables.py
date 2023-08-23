import os
from typing import Optional

RUN_ID_PREFIX = "RUN_"


class EnvironmentVariables:
    """Class to dynamically get any environment variables."""

    # Infra

    @property
    def WANDB_BASE_URL(self) -> str:
        return os.getenv("WANDB_BASE_URL")

    @property
    def AWS_PROFILE(self) -> str:
        return os.getenv("AWS_PROFILE")

    # DDP
    @property
    def LOCAL_RANK(self) -> int:
        return int(os.getenv("LOCAL_RANK", -1))

    # Turn ON/OFF features
    @property
    def CRASH_HANDLER(self) -> str:
        return os.getenv("CRASH_HANDLER", "TRUE")

    @property
    def UPLOAD_LOGS(self) -> bool:
        return os.getenv("UPLOAD_LOGS", "TRUE") == "TRUE"

    @property
    def FILE_LOG_LEVEL(self) -> str:
        return os.getenv("FILE_LOG_LEVEL", default="DEBUG").upper()

    @property
    def CONSOLE_LOG_LEVEL(self) -> str:
        return os.getenv("CONSOLE_LOG_LEVEL", default="INFO").upper()

    @property
    def HYDRA_FULL_ERROR(self) -> Optional[str]:
        return os.getenv("HYDRA_FULL_ERROR")

    @HYDRA_FULL_ERROR.setter
    def HYDRA_FULL_ERROR(self, value: str):
        os.environ["HYDRA_FULL_ERROR"] = value

    @property
    def DDP_RUN_ID(self) -> Optional[str]:
        """
        Retrieve the Distributed Data Parallel (DDP) run identifier.

        If `DDP_RUN_ID` is not explicitly set, this property returns an identifier that's used for all subprocesses during a DDP operation.
        This helps in maintaining consistency across various subprocesses.
        Specifically, if `TORCHELASTIC_RUN_ID` is defined in any DDP subprocesses and `DDP_RUN_ID` was not set,
        the value of `TORCHELASTIC_RUN_ID` will be used as `DDP_RUN_ID`.
        """
        return os.environ.get("TORCHELASTIC_RUN_ID")


env_variables = EnvironmentVariables()
