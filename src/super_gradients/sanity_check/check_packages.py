import pkg_resources
from typing import List, Union, Optional
from pathlib import Path
from packaging.requirements import Requirement

from .display_utils import format_error_msg

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__, "DEBUG")


__all__ = ["check_packages"]


def _get_requirements_path(requirements_file_name: str) -> Union[None, Path]:
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
    file_path = Path(__file__)  # super-gradients/src/super_gradients/sanity_check/env_sanity_check.py
    package_root = file_path.parent.parent  # moving to super-gradients/src/super_gradients
    project_root = package_root.parent.parent  # moving to super-gradients

    # If installed from artifact, requirements.txt is in package_root, if installed locally it is in project_root
    if (package_root / requirements_file_name).exists():
        return package_root / requirements_file_name
    elif (project_root / requirements_file_name).exists():
        return project_root / requirements_file_name
    else:
        return None  # Could happen when installed through github directly ("pip install git+https://github.com/...")


def _get_requirements(use_pro_requirements: bool) -> Optional[List[str]]:
    requirements_path = _get_requirements_path("requirements.txt")
    pro_requirements_path = _get_requirements_path("requirements.pro.txt")

    if requirements_path is None:
        return None

    with open(requirements_path, "r") as f:
        requirements = f.read().splitlines()

    with open(pro_requirements_path, "r") as f:
        pro_requirements = f.read().splitlines()

    valid_requirements = filter(
        lambda x: x and x[0].isalpha(),
        requirements + pro_requirements if use_pro_requirements else requirements,
    )
    return list(valid_requirements)


def check_packages(test_name: str):
    """Check that all installed libs respect the requirement.txt and requirements.pro.txt if relevant.

    :param test_name: Name that is used to refer to this test.
    :return: True if test was successful, False otherwise
    """

    installed_packages = {package.key.lower(): package.version for package in pkg_resources.working_set}
    requirements = _get_requirements(use_pro_requirements="deci-lab-client" in installed_packages)

    if requirements is None:
        logger.info(msg='Library check is not supported when super_gradients installed through "git+https://github.com/..." command')
        return

    for requirement in requirements:
        req = Requirement(requirement)
        package_name, package_spec = req.name.lower(), req.specifier

        if package_name not in installed_packages.keys():
            error = f"{package_name} required but not found"
            logger.error(msg=format_error_msg(test_name=test_name, error_msg=error))
        else:
            installed_version = installed_packages[package_name]
            if installed_version not in package_spec:
                error = f"{package_name}=={installed_version} does not satisfy requirement ({requirement})"
                logger.error(msg=format_error_msg(test_name=test_name, error_msg=error))
