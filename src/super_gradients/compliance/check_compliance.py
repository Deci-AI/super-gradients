"""
A Python script that check's the compliance of Hosts (OS and Hardware) for Deci's infrastructure.
"""

import os
import sys
from pip._internal.operations.freeze import freeze
from logging import getLogger, DEBUG
from typing import List, Dict
from pathlib import Path
from packaging.version import Version

logger = getLogger('sg-compliance')
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
    """Read requirement.txt from the root, and split it as a list of libs/version"""
    file_path = Path(__file__)  # super-gradients/src/super_gradients/compliance/check_compliance.py
    project_root = file_path.parent.parent.parent.parent  # moving to super-gradients, where requirements.txt is
    with open(project_root / "requirements.txt", "r") as f:
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


def check_compliance() -> None:
    """Run all the compliance test and log everything that does not meet requirements"""

    requirement_checkers = {
        'operating_system': verify_os,
        'libraries': verify_installed_libraries,
    }

    compliance_errors = {}
    logger.info('SuperGradients Compliance Check Started')
    logger.info(f'Checking the following components: {list(requirement_checkers.keys())}')
    logger.info('_' * 20)
    for test_name, test_function in requirement_checkers.items():
        logger.info(f"Verifying {test_name}...")
        try:
            errors = test_function()
            if len(errors) > 0:
                compliance_errors[test_name] = errors
                for e in errors:
                    assert isinstance(e, str), 'Errors should be returned by the functions as str objects.'
                    print_error(test_name, e)
            else:
                logger.info(f'{test_name} OK')
            logger.info('_' * 20)
        except Exception as e:
            logger.fatal(f'Failed to check for compliance: {e}', exc_info=True)
            raise

    if compliance_errors:
        logger.fatal(
            f'The current environment does not meet Deci\'s needs, errors found in: {", ".join(list(compliance_errors.keys()))}')
    else:
        logger.info('Great, Looks like the current environment meet\'s Deci\'s requirements!')


if __name__ == '__main__':
    check_compliance()
