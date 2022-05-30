"""
A Python script that check's the compliance of Hosts (OS and Hardware) for Deci's infrastructure.
"""

import os
import platform
import subprocess
import sys
from logging import getLogger, StreamHandler, DEBUG
from math import e as math_e
from sys import stdout
from typing import List, Dict

from pathlib import Path
import GPUtil
import psutil
from pip._internal.operations.freeze import freeze
from packaging.version import Version

logger = None


def get_requirements() -> List[str]:
    """Read requirement.txt from the root, and split it as a list of libs/version"""
    with open(Path(__file__).parent.parent.parent.parent / "requirements.txt", "r") as f:
        requirements_str = f.read()
    return requirements_str.split("\n")


def get_installed_libs_with_version() -> Dict[str, str]:
    """Get all the installed libraries, and outputs it as a dict: lib -> version"""
    installed_libs_with_version = {}
    for lib_with_version in freeze():
        if "==" in lib_with_version:
            lib, version = lib_with_version.split("==")
            installed_libs_with_version[lib.lower()] = version
    return installed_libs_with_version


# class VersionMismatchError(Exception):
#     """Exception to raise when the version of an installed lib does not match requirement"""
#     __module__ = Exception.__module__


def verify_installed_libraries() -> None:
    """Check that all installed libs respect the requirement.txt"""
    requirements = get_requirements()
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


def verify_os() -> List[str]:
    """Verifying operating system name and platform"""
    name = os.name
    platform = sys.platform
    logger.info(f'OS Name: {name}')
    logger.info(f'OS Platform: {platform}')
    if 'linux' not in platform.lower():
        return ['Deci officially supports only Linux kernels. Some features may not work as expected.']
    return []


def print_error(component_name, error):
    error_message = f"Failed to verify {component_name}: {error}"
    logger.error(error_message)


def check_host_compliance():
    #

    # try:
    #     print(lib)
    #     module = import_module(lib)
    #     if isinstance(module.__version__, tuple):
    #         installed_version = Version(".".join(map(str, module.__version__)))
    #     else:
    #         installed_version = Version(module.__version__)
    #
    #     required_version = Version(required_version)
    #
    #     if constraint == ">=":
    #         if not installed_version >= required_version:
    #             raise ImportError(f"{module.__name__} is installed with version {installed_version.version} < {required_version.version}")
    #
    #     if constraint == "~=":
    #         if not (installed_version.major == required_version.major):
    #             raise ImportError(
    #                 f"{module.__name__} is installed with major version {installed_version.version.major} != {required_version.major}")
    #         if not (installed_version.minor >= required_version.minor):
    #             raise ImportError(
    #                 f"{module.__name__} is installed with minor version {installed_version.version.minor} < {required_version.minor}")
    #
    #     if constraint == "==":
    #         if not installed_version == required_version:
    #             raise ImportError(f"{module.__name__} is installed with version {installed_version.version} != {required_version.version}")
    # except ModuleNotFoundError:
    #     print(f"{requirement} NOT FOUND")

    # required_libs_grp_by_constraint

    requirement_checkers = {
        'operating_system': verify_os,
        'libraries': verify_installed_libraries,
    }

    # Adding a logger
    logger = getLogger('deci-compliance')
    logger.addHandler(StreamHandler(stream=stdout))
    logger.setLevel(DEBUG)

    compliance_errors = {}
    logger.info('Deci Compliance Check Started')
    logger.info(f'Checking the following components: {list(requirement_checkers.keys())}')
    for test_name, test_function in requirement_checkers.items():
        logger.info(f"Verifying {test_name}...")
        try:
            errors = test_function()
            if errors:
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
    check_host_compliance()