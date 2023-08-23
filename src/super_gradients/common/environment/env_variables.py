import os
from typing import Optional


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
    def RUN_ID(self) -> Optional[str]:
        if os.environ.get("RUN_ID"):
            return os.environ["RUN_ID"]
        else:
            # `TORCHELASTIC_RUN_ID` is defined in DDP subprocesses
            # If `RUN_ID` was not set explicitly, and we use DDP (`TORCHELASTIC_RUN_ID` is not None), then we want to use it.
            # This ensures that the same `RUN_ID` will be used for all subprocesses.
            return os.environ.get("TORCHELASTIC_RUN_ID")

    @RUN_ID.setter
    def RUN_ID(self, value: str):
        os.environ["RUN_ID"] = value


env_variables = EnvironmentVariables()
