import os


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


env_variables = EnvironmentVariables()
