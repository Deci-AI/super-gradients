import pkg_resources
from typing import Dict


def get_installed_packages() -> Dict[str, str]:
    """Map all the installed packages to their version."""
    return {package.key.lower(): package.version for package in pkg_resources.working_set}
