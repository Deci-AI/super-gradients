import sys
import pkg_resources
from pkg_resources import parse_version
from packaging.specifiers import SpecifierSet
from typing import List, Optional
from pathlib import Path


from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.ddp_utils import is_main_process

logger = get_logger(__name__, "DEBUG")


def format_error_msg(test_name: str, error_msg: str) -> str:
    """Format an error message in the appropriate format.

    :param test_name:   Name of the test being tested.
    :param error_msg:   Message to format in appropriate format.
    :return:            Formatted message
    """
    return f"\33[31mFailed to verify {test_name}: {error_msg}\33[0m"


def check_os():
    """Check the operating system name and platform."""

    if "linux" not in sys.platform.lower():
        error = "Deci officially supports only Linux kernels. Some features may not work as expected."
        logger.warning(msg=format_error_msg(test_name="operating system", error_msg=error))


def get_requirements_path(requirements_file_name: str) -> Optional[Path]:
    """Get the path of requirement.txt from the root if exist.
    There is a difference when installed from artifact or locally.
        - In the first case, requirements.txt is copied to the package during the CI.
        - In the second case, requirements.txt in the root of the project.

    Note: This is because when installed from artifact only the source code is accessible, so requirements.txt has to be
          copied to the package root (./src/super_gradients). This is automatically done with the CI to make sure that
          in the github we only have 1 source of truth for requirements.txt. The consequence being that when the code
          is copied/cloned from github, the requirements.txt was not copied to the super_gradients package root, so we
          need to go to the project root (.) to find it.
    """
    file_path = Path(__file__)  # Refers to: .../super-gradients/src/super_gradients/sanity_check/env_sanity_check.py
    package_root = file_path.parent.parent  # Refers to: .../super-gradients/src/super_gradients
    project_root = package_root.parent.parent  # Refers to .../super-gradients

    # If installed from artifact, requirements.txt is in package_root, if installed locally it is in project_root
    if (package_root / requirements_file_name).exists():
        return package_root / requirements_file_name
    elif (project_root / requirements_file_name).exists():
        return project_root / requirements_file_name
    else:
        return None  # Could happen when installed through github directly ("pip install git+https://github.com/...")


def get_requirements(use_pro_requirements: bool) -> Optional[List[str]]:
    requirements_path = get_requirements_path("requirements.txt")
    pro_requirements_path = get_requirements_path("requirements.pro.txt")

    if (requirements_path is None) or (pro_requirements_path is None):
        return None

    with open(requirements_path, "r") as f:
        requirements = f.read().splitlines()

    with open(pro_requirements_path, "r") as f:
        pro_requirements = f.read().splitlines()

    return requirements + pro_requirements if use_pro_requirements else requirements


def check_packages():
    """Check that all installed libs respect the requirement.txt, and requirements.pro.txt if relevant.
    Note: We only log an error
    """
    test_name = "installed packages"

    installed_packages = {package.key.lower(): package.version for package in pkg_resources.working_set}
    requirements = get_requirements(use_pro_requirements="deci-platform-client" in installed_packages)

    if requirements is None:
        logger.info(msg='Library check is not supported when super_gradients installed through "git+https://github.com/..." command')
        return

    for requirement in pkg_resources.parse_requirements(requirements):
        package_name = requirement.name.lower()

        if package_name not in installed_packages.keys():
            error = f"{package_name} required but not found"
            logger.warning(msg=format_error_msg(test_name=test_name, error_msg=error))
            continue

        installed_version_str = installed_packages[package_name]
        for operator_str, req_version_str in requirement.specs:

            installed_version = parse_version(installed_version_str)
            req_version = parse_version(req_version_str)
            req_spec = SpecifierSet(operator_str + req_version_str)

            if installed_version_str not in req_spec:
                error = f"{package_name}=={installed_version} does not satisfy requirement {requirement}"

                requires_at_least = operator_str in ("==", "~=", ">=", ">")
                if requires_at_least and installed_version < req_version:
                    logger.warning(msg=format_error_msg(test_name=test_name, error_msg=error))
                else:
                    logger.debug(msg=error)


def env_sanity_check():
    """Run the sanity check tests and log everything that does not meet requirements."""
    if is_main_process():
        check_os()
        check_packages()


if __name__ == "__main__":
    env_sanity_check()
