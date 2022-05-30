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
from typing import List

import GPUtil
import psutil
from pip._internal.operations.freeze import freeze

logger = None


def verify_cpu() -> List[str]:
    # gives a single float value
    logical_cpus_count = psutil.cpu_count()
    physical_cpus_count = psutil.cpu_count(logical=False)

    cpu_name = platform.processor()

    logger.info(f"Found {physical_cpus_count} CPUs with {logical_cpus_count} cores: {cpu_name}")
    return []


def verify_gpu() -> List[str]:
    all_gpus = GPUtil.getGPUs()
    for gpu in all_gpus:
        logger.info(
            f'Found GPU: {gpu.name} {gpu.serial} {gpu.driver}, Total Memory: {gpu.memoryTotal}, Free: {gpu.memoryFree}')

    available_gpus = GPUtil.getAvailable()
    if available_gpus:
        logger.info(f'Found {len(available_gpus)} available GPUs')
        used_gpus_count = len(all_gpus) - len(available_gpus)
        if used_gpus_count != 0:
            logger.warning(f'Seems like {used_gpus_count} GPUs are currently in use.')
        return []
    return ["No available GPUs."]


def verify_memory() -> List[str]:
    memory = psutil.virtual_memory()
    required_free_gb = 2
    minimum_free_memory_bytes = (required_free_gb * math_e + 9)
    if minimum_free_memory_bytes >= memory.total:
        return [
            f"Not enough free memory - At most {required_free_gb}GB must be free"]
    return []


def verify_installed_libraries() -> List[str]:
    installed_libraries = list(freeze())
    for lib in installed_libraries:
        logger.info(f"{lib}")
    logger.info(f'Found {len(installed_libraries)} installed libraries')
    return []


def verify_docker_host() -> List[str]:
    try:
        docker_version_text = subprocess.check_output(["docker", "version"]).decode()
    except FileNotFoundError:
        return ["Docker doesn't seem to be installed."]

    if 'Server: Docker Engine' in docker_version_text:
        logger.info('Docker version:' + docker_version_text)
        return []
    return ["Docker doesn't seem to be installed."]


def verify_ssh_server() -> List[str]:
    error = 'An SSH Server (sshd) does not seem to be installed. We recommend using the latest OpenSSH Server. ' \
            'If an SSH Server is installed, you may ignore this notice, or change the compliance check configuration.'
    try:
        ssh_server_command_output = subprocess.check_output(["which", "sshd"]).decode()
    except subprocess.CalledProcessError:
        return [error]
    if '/sbin/sshd' not in ssh_server_command_output:
        return [error]
    return []


def verify_os() -> List[str]:
    # Verifying operating system name and platform
    name = os.name
    platform = sys.platform
    logger.info(f'OS Name: {name}')
    logger.info(f'OS Platform: {platform}')
    if 'linux' not in platform.lower():
        return ['Deci officially supports only Linux kernels. Some features may not work as expected.']
    return []


# Memory and utilization of GPUs (maybe some of them already in use and we'll get OOM).
# CPU processor type. might be relevant for CPU inference. Also the number of cores is relevant for training.

def print_error(component_name, error):
    error_message = f"Failed to verify {component_name}: {error}"
    logger.error(error_message)


if __name__ == '__main__':
    from pathlib import Path
    print(Path(__file__))
    print(Path(__file__).parent)
    with open(Path(__file__).parent.parent / "requirements.txt", "r") as f:
        requirements = f.read().split("\n")

    #
    # required_libs_grp_by_constraint = {">=": {}, "~=": {}, "==": {}}
    # require_greater_version_libs = {}
    # require_compatible_version_libs = {}
    # require_exact_version_libs = {}
    # unspecified_requirements = []
    #
    #
    installed_lib_with_versions = {}
    for lib_with_version in freeze():
        if "==" in lib_with_version:
            lib, version = lib_with_version.split("==")
            installed_lib_with_versions[lib.lower()] = version
    #
    from packaging.version import Version, parse
    from importlib import import_module

    for requirement in requirements:
        if ">=" in requirement:
            constraint = ">="
        elif "~=" in requirement:
            constraint = "~="
        elif "==" in requirement:
            constraint = "=="
        else:
            continue

        lib, required_version = requirement.split(constraint)

        if lib.lower() not in installed_lib_with_versions:
            raise ImportError(f"{lib} required but not found")
        installed_version = installed_lib_with_versions[lib.lower()]

        installed_version = Version(installed_version)
        required_version = Version(required_version)

        error_msg = f"{lib} is installed with version {installed_version} which does not satisfy {requirement}"
        if constraint == ">=":
            if not installed_version >= required_version:
                raise ImportError(error_msg)
        elif constraint == "~=":
            if not (installed_version.major == required_version.major and installed_version.minor == required_version.minor and installed_version.micro >= required_version.micro):
                raise ImportError(error_msg)
        elif constraint == "==":
            if not installed_version == required_version:
                raise ImportError(error_msg)

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

    # requirement_checkers = {
    #     # 'operating_system': verify_os,
    #     'cpu': verify_cpu,
    #     'gpu': verify_gpu,
    #     'memory': verify_memory,
    #     'libraries': verify_installed_libraries,
    #     # 'docker': verify_docker_host,
    #     # 'ssh': verify_ssh_server,
    # }
    #
    # # Adding a logger
    # logger = getLogger('deci-compliance')
    # logger.addHandler(StreamHandler(stream=stdout))
    # logger.setLevel(DEBUG)
    #
    # compliance_errors = {}
    # logger.info('Deci Compliance Check Started')
    # logger.info(f'Checking the following components: {list(requirement_checkers.keys())}')
    # for test_name, test_function in requirement_checkers.items():
    #     logger.info(f"Verifying {test_name}...")
    #     try:
    #         errors = test_function()
    #         if errors:
    #             compliance_errors[test_name] = errors
    #             for e in errors:
    #                 assert isinstance(e, str), 'Errors should be returned by the functions as str objects.'
    #                 print_error(test_name, e)
    #         else:
    #             logger.info(f'{test_name} OK')
    #         logger.info('_' * 20)
    #     except Exception as e:
    #         logger.fatal(f'Failed to check for compliance: {e}', exc_info=True)
    #         raise
    #
    # if compliance_errors:
    #     logger.fatal(
    #         f'The current environment does not meet Deci\'s needs, errors found in: {", ".join(list(compliance_errors.keys()))}')
    # else:
    #     logger.info('Great, Looks like the current environment meet\'s Deci\'s requirements!')