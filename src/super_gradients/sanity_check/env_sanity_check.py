import logging
import os
import sys
from pip._internal.operations.freeze import freeze
from typing import List, Dict, Union
from pathlib import Path
from packaging.version import Version

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.ddp_utils import is_main_process

LIB_CHECK_IMPOSSIBLE_MSG = 'Library check is not supported when super_gradients installed through "git+https://github.com/..." command'

logger = get_logger(__name__, log_level=logging.DEBUG)


def get_requirements_path(requirements_file_name: str) -> Union[None, Path]:
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

    requirements_path = get_requirements_path("requirements.txt")
    pro_requirements_path = get_requirements_path("requirements.pro.txt")

    if requirements_path is None:
        return [LIB_CHECK_IMPOSSIBLE_MSG]

    with open(requirements_path, "r") as f:
        requirements = f.readlines()

    installed_libs_with_version = get_installed_libs_with_version()

    # if pro_requirements_path is not None:
    with open(pro_requirements_path, "r") as f:
        pro_requirements = f.readlines()
    if "deci-lab-client" in installed_libs_with_version:
        requirements += pro_requirements

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

        if ",<=" in required_version_str:
            upper_limit_version = Version(required_version_str.split(",<=")[1])
            required_version_str = required_version_str.split(",<=")[0]
            constraint += ",<="

        if lib.lower() not in installed_libs_with_version.keys():
            errors.append(f"{lib} required but not found")
            continue

        installed_version_str = installed_libs_with_version[lib.lower()]
        installed_version, required_version = Version(installed_version_str), Version(required_version_str)

        is_constraint_respected = {
            ">=,<=": required_version <= installed_version <= upper_limit_version,
            ">=": installed_version >= required_version,
            "~=": (
                installed_version.major == required_version.major
                and installed_version.minor == required_version.minor
                and installed_version.micro >= required_version.micro
            ),
            "==": installed_version == required_version,
        }
        if not is_constraint_respected[constraint]:
            errors.append(f"{lib} is installed with version {installed_version} which does not satisfy {requirement} (based on {requirements_path})")
    return errors


def verify_os() -> List[str]:
    """Verifying operating system name and platform"""
    if "linux" not in sys.platform.lower():
        return ["Deci officially supports only Linux kernels. Some features may not work as expected."]
    return []


def run_env_sanity_check():
    """Run the sanity check tests and log everything that does not meet requirements"""

    display_sanity_check = os.getenv("DISPLAY_SANITY_CHECK", "False") == "True"
    stdout_log_level = logging.INFO if display_sanity_check else logging.DEBUG

    logger.setLevel(logging.DEBUG)  # We want to log everything regardless of DISPLAY_SANITY_CHECK

    requirement_checkers = {
        "operating_system": verify_os,
        "libraries": verify_installed_libraries,
    }

    logger.log(stdout_log_level, "SuperGradients Sanity Check Started")
    logger.log(stdout_log_level, f"Checking the following components: {list(requirement_checkers.keys())}")
    logger.log(stdout_log_level, "_" * 20)

    lib_check_is_impossible = False
    sanity_check_errors = {}
    for test_name, test_function in requirement_checkers.items():
        logger.log(stdout_log_level, f"Verifying {test_name}...")

        errors = test_function()
        if errors == [LIB_CHECK_IMPOSSIBLE_MSG]:
            lib_check_is_impossible = True
            logger.log(stdout_log_level, LIB_CHECK_IMPOSSIBLE_MSG)
        elif len(errors) > 0:
            sanity_check_errors[test_name] = errors
            for error in errors:
                logger.log(logging.ERROR, f"\33[31mFailed to verify {test_name}: {error}\33[0m")
        else:
            logger.log(stdout_log_level, f"{test_name} OK")
        logger.log(stdout_log_level, "_" * 20)

    if sanity_check_errors:
        logger.log(stdout_log_level, f'The current environment does not meet Deci\'s needs, errors found in: {", ".join(list(sanity_check_errors.keys()))}')
    elif lib_check_is_impossible:
        logger.log(stdout_log_level, LIB_CHECK_IMPOSSIBLE_MSG)
    else:
        logger.log(stdout_log_level, "Great, Looks like the current environment meet's Deci's requirements!")

    # The last message needs to be displayed independently of DISPLAY_SANITY_CHECK
    if display_sanity_check:
        logger.info("** This check can be hidden by setting the env variable DISPLAY_SANITY_CHECK=False prior to import. **")
    else:
        logger.info(
            "** A sanity check is done when importing super_gradients for the first time. ** "
            "-> You can see the details by setting the env variable DISPLAY_SANITY_CHECK=True prior to import."
        )


def env_sanity_check():
    """Run the sanity check tests and log everything that does not meet requirements"""
    if is_main_process():
        run_env_sanity_check()


if __name__ == "__main__":
    env_sanity_check()
