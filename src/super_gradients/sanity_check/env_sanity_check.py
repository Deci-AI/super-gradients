import os
import sys
from pip._internal.operations.freeze import freeze
from logging import getLogger, DEBUG
from typing import List, Dict, Union, Tuple
from pathlib import Path
from packaging.version import Version

logger = getLogger('sg-sanity-check')
logger.setLevel(DEBUG)

LIB_CHECK_IMPOSSIBLE_MSG = 'Library check is not supported when super_gradients installed through "git+https://github.com/..." command'


def verify_os() -> List[str]:
    """Verifying operating system name and platform"""
    name = os.name
    platform = sys.platform
    logger.info(f'OS Name: {name}')
    logger.info(f'OS Platform: {platform}')
    if 'linux' not in platform.lower():
        return ['Deci officially supports only Linux kernels. Some features may not work as expected.']
    return []


PACKAGE_REQUIREMENT = "package_requirement"
PROJECT_REQUIREMENT = "project_requirement"
NO_REQUIREMENT = "no_requirement"


def get_requirements_path() -> Union[None, Path]:
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
    if (package_root / "requirements.txt").exists():
        print("pack")
        return package_root / "requirements.txt"
    elif (project_root / "requirements.txt").exists():
        print("project")
        return project_root / "requirements.txt"
    else:
        return None  # Could happen when installed through github directly ("pip install git+https://github.com/...")


def get_installed_libs_with_version() -> Dict[str, str]:
    """Get all the installed libraries, and outputs it as a dict: lib -> version"""
    installed_libs_with_version = {}
    for lib_with_version in freeze():
        if "==" in lib_with_version:
            lib, version = lib_with_version.split("==")
            installed_libs_with_version[lib.lower()] = version
    return installed_libs_with_version


def verify_installed_libraries() -> List[str]:
    """Check that all installed libs respect the requirement.txt"""

    requirements_path = get_requirements_path()
    if requirements_path is None:
        return [LIB_CHECK_IMPOSSIBLE_MSG]

    with open(requirements_path, "r") as f:
        requirements = f.readlines()

    installed_libs_with_version = get_installed_libs_with_version()

    errors = []
    for requirement in requirements:
        if ">=" in requirement:
            constraint = ">="
        elif "~=" in requirement:
            constraint = "~="
        elif "==" in requirement:
            constraint = "=="
        else:
            continue

        lib, required_version_str = requirement.split(constraint)

        if lib.lower() not in installed_libs_with_version.keys():
            errors.append(f"{lib} required but not found")
            continue

        installed_version_str = installed_libs_with_version[lib.lower()]
        installed_version, required_version = Version(installed_version_str), Version(required_version_str)

        is_constraint_respected = {
            ">=": installed_version >= required_version,
            "~=": installed_version.major == required_version.major and installed_version.minor == required_version.minor and installed_version.micro >= required_version.micro,
            "==": installed_version == required_version
        }
        if not is_constraint_respected[constraint]:
            errors.append(
                f"{lib} is installed with version {installed_version} which does not satisfy {requirement} (based on {requirements_path})")
    return errors


def print_error(component_name: str, error: str) -> None:
    error_message = f"Failed to verify {component_name}: {error}"
    logger.warning(error_message)


def env_sanity_check() -> None:
    """Run the sanity check tests and log everything that does not meet requirements"""

    requirement_checkers = {
        'operating_system': verify_os,
        'libraries': verify_installed_libraries,
    }

    logger.info('SuperGradients Sanity Check Started')
    logger.info(f'Checking the following components: {list(requirement_checkers.keys())}')
    logger.info('_' * 20)

    lib_check_is_impossible = False
    sanity_check_errors = {}
    for test_name, test_function in requirement_checkers.items():
        logger.info(f"Verifying {test_name}...")

        errors = test_function()
        if errors == [LIB_CHECK_IMPOSSIBLE_MSG]:
            lib_check_is_impossible = True
            logger.warning(LIB_CHECK_IMPOSSIBLE_MSG)
        elif len(errors) > 0:
            sanity_check_errors[test_name] = errors
            for error in errors:
                print_error(test_name, error)
        else:
            logger.info(f'{test_name} OK')
        logger.info('_' * 20)

    if sanity_check_errors:
        logger.warning(
            f'The current environment does not meet Deci\'s needs, errors found in: {", ".join(list(sanity_check_errors.keys()))}')
    elif lib_check_is_impossible:
        logger.warning(LIB_CHECK_IMPOSSIBLE_MSG)
    else:
        logger.info('Great, Looks like the current environment meet\'s Deci\'s requirements!')


if __name__ == '__main__':
    env_sanity_check()
