import os


class EnvironmentVariables:
    """Class to dynamically get any environment variables."""

    # Infra
    @property
    def DECI_PLATFORM_TOKEN(self):
        return os.getenv("DECI_PLATFORM_TOKEN")

    @property
    def WANDB_BASE_URL(self):
        return os.getenv("WANDB_BASE_URL")

    @property
    def AWS_PROFILE(self):
        return os.getenv("AWS_PROFILE")

    # DDP
    @property
    def LOCAL_RANK(self):
        return int(os.getenv("LOCAL_RANK", -1))

    # Turn ON/OFF features
    @property
    def CRASH_HANDLER(self):
        return os.getenv("CRASH_HANDLER", "TRUE")

    @property
    def UPLOAD_LOGS(self):
        return os.getenv("UPLOAD_LOGS", "TRUE")

    @property
    def DECI_API_HOST(self) -> str:
        return os.getenv("DECI_API_HOST", default="api.deci.ai")


env_variables = EnvironmentVariables()
