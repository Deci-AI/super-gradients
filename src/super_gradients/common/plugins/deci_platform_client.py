from super_gradients.common import env_variables

client_enabled = True
try:
    from deci_lab_client.client import DeciPlatformClient
except (ImportError, NameError):
    client_enabled = False


def instantiate_deci_platform_client() -> DeciPlatformClient:
    """Instantiate DeciPlatformClient on prod/dev relatively to the env requirements.

    :param api_port: The port of deci's platform HTTP API.
    :param https: Whether to use https instead of HTTP. Using https Will add latency.
    """
    if client_enabled:
        api_host = "api.deci.ai" if env_variables.PROD_ENVIRONMENT else "api.development.deci.ai"
        print(api_host)
        client = DeciPlatformClient(api_host=api_host)
        client.login(token=env_variables.DECI_PLATFORM_TOKEN)
        return client
    else:
        raise RuntimeError("Trying to instantiate deci_lab_client.client.DeciPlatformClient but deci_lab_client not installed...")
