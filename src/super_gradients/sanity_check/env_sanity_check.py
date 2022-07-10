import os
import sys
from pip._internal.operations.freeze import freeze
from logging import getLogger, DEBUG
from typing import List, Dict
from pathlib import Path
from packaging.version import Version

logger = getLogger('sg-sanity-check')
logger.setLevel(DEBUG)


def verify_os() -> List[str]:
    """Verifying operating system name and platform"""
    name = os.name
    platform = sys.platform
    logger.info(f'OS Name: {name}')
    logger.info(f'OS Platform: {platform}')
    if 'linux' not in platform.lower():
        return ['Deci officially supports only Linux kernels. Some features may not work as expected.']
    return []


def get_libs_requirements() -> List[str]:
    """Read requirement.txt from the root, and split it as a list of libs/version.
    There is a difference when installed from artefact or locally.
        - In the first case, requirements.txt is copied to the package during the CI.
        - In the second case, requirements.txt in the root of the project.

    Note: This is because when installed from artefact only the source code is accessible, so requirements.txt has to be
          copied to the package root (./src/super_gradients). This is automatically done with the CI to make sure that
          in the github we only have 1 source of truth for requirements.txt. The consequence being that when the code
          is copied/cloned from github, the requirements.txt was not copied to the super_gradients package root, so we
          need to go to the project root (.) to find it.
    """
    file_path = Path(__file__)  # super-gradients/src/super_gradients/sanity_check/env_sanity_check.py
    package_root = file_path.parent.parent  # moving to super-gradients/src/super_gradients
    project_root = package_root.parent.parent  # moving to super-gradients

    # If installed from artefact, requirements.txt is in package_root, if installed locally it is in project_root
    requirements_folder = package_root if (package_root / "requirements.txt").exists() else project_root
    with open(requirements_folder / "requirements.txt", "r") as f:
        return f.readlines()


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
    requirements = get_libs_requirements()
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
                f"{lib} is installed with version {installed_version} which does not satisfy {requirement}")
    return errors


def print_error(component_name: str, error: str) -> None:
    error_message = f"Failed to verify {component_name}: {error}"
    logger.error(error_message)


def env_sanity_check() -> None:
    """Run the sanity check tests and log everything that does not meet requirements"""

    requirement_checkers = {
        'operating_system': verify_os,
        'libraries': verify_installed_libraries,
    }

    sanity_check_errors = {}
    logger.info('SuperGradients Sanity Check Started')
    logger.info(f'Checking the following components: {list(requirement_checkers.keys())}')
    logger.info('_' * 20)
    for test_name, test_function in requirement_checkers.items():
        logger.info(f"Verifying {test_name}...")
        try:
            errors = test_function()
            if len(errors) > 0:
                sanity_check_errors[test_name] = errors
                for e in errors:
                    assert isinstance(e, str), 'Errors should be returned by the functions as str objects.'
                    print_error(test_name, e)
            else:
                logger.info(f'{test_name} OK')
            logger.info('_' * 20)
        except Exception as e:
            logger.fatal(f'Failed to check for sanity_check: {e}', exc_info=True)
            raise

    if sanity_check_errors:
        logger.fatal(
            f'The current environment does not meet Deci\'s needs, errors found in: {", ".join(list(sanity_check_errors.keys()))}')
    else:
        logger.info('Great, Looks like the current environment meet\'s Deci\'s requirements!')


if __name__ == '__main__':
    env_sanity_check()
