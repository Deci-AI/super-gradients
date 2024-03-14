import subprocess
import sys
from typing import Optional


def install_package(package: str, extra_index_url: Optional[str] = None) -> None:
    """
    Install a package using pip. The function will raise an exception if the installation fails.

    :param package: A package name (Can be with version)
    :param extra_index_url: Optional extra index url
    :return: None
    """
    command = [sys.executable, "-m", "pip", "install", package]
    if extra_index_url:
        command.extend(["--extra-index-url", extra_index_url])
    subprocess.check_call(command)
