import os
import sys
import pkg_resources
from typing import Optional
from datetime import datetime

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.ddp_utils import execute_and_distribute_from_master


try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    PKG_CHECKPOINTS_DIR = None


logger = get_logger(__name__)


@execute_and_distribute_from_master
def generate_run_id() -> str:
    """Generate a unique run ID based on the current timestamp.

    :return: Unique run ID. in the format "RUN_<year><month><day>_<hour><minute><second>_<microseconds>" (E.g. "RUN_20230802_131052_651906")
    """
    return datetime.now().strftime("RUN_%Y%m%d_%H%M%S_%f")


def is_run_dir(dirname: str) -> bool:
    """Check if a directory is a run directory.

    :param dirname: Directory name.
    :return:        True if the directory is a run directory, False otherwise.
    """
    return os.path.basename(dirname).startswith("RUN_")


def get_latest_run_id(experiment_name: str, checkpoints_root_dir: Optional[str] = None) -> Optional[str]:
    """
    :param experiment_name:         Name of the experiment.
    :param checkpoints_root_dir:    Path to the directory where all the experiments are organised, each sub-folder representing a specific experiment.
                                    If None, SG will first check if a package named 'checkpoints' exists.
                                    If not, SG will look for the root of the project that includes the script that was launched.
                                    If not found, raise an error.
    :return:                        Latest valid run ID. in the format "RUN_<year>"
    """
    experiment_dir = get_experiment_dir_path(checkpoints_root_dir=checkpoints_root_dir, experiment_name=experiment_name)

    run_dirs = [os.path.join(experiment_dir, folder) for folder in os.listdir(experiment_dir) if is_run_dir(folder)]
    for run_dir in sorted(run_dirs, reverse=True):
        if "ckpt_latest.pth" not in os.listdir(run_dir):
            logger.warning(
                f"Latest run directory {run_dir} does not contain a `ckpt_latest.pth` file, so it cannot be resumed. "
                f"Trying to load the n-1 most recent run..."
            )
        else:
            return os.path.basename(run_dir)


def validate_run_id(run_id: str, experiment_name: str, ckpt_root_dir: Optional[str] = None):
    experiment_dir = get_experiment_dir_path(checkpoints_root_dir=ckpt_root_dir, experiment_name=experiment_name)
    run_dir = os.path.join(experiment_dir, run_id)
    if not os.path.exists(run_dir) and not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f'Invalid run directory "{run_dir}", with `ckpt_root_dir={ckpt_root_dir}`, `experiment_name={experiment_name}`, `run_dir={run_dir}`.'
        )


def _get_project_root_path() -> Optional[str]:
    """Extract the path of first project that includes the script that was launched. Return None if no project found."""
    script_path = os.path.abspath(path=sys.argv[0])
    return _parse_project_root_path(path=os.path.dirname(script_path))


def _parse_project_root_path(path: str) -> Optional[str]:
    """Extract the path of first project that includes this path (recursively look into parent folders). Return None if no project found."""
    if path in ("", "/"):
        return None
    is_project_root_path = any(os.path.exists(os.path.join(path, file)) for file in (".git", "requirements.txt", ".env", "venv", "setup.py"))
    return path if is_project_root_path else _parse_project_root_path(path=os.path.dirname(path))


def get_project_checkpoints_dir_path() -> Optional[str]:
    """Get the checkpoints' directory that is at the root of the users project. Create it if it doesn't exist. Return None if root not found."""
    project_root_path = _get_project_root_path()
    if project_root_path is None:
        return None

    checkpoints_path = os.path.join(project_root_path, "checkpoints")
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        logger.info(f'A checkpoints directory was just created at "{checkpoints_path}". To work with another directory, please set "ckpt_root_dir"')
    return checkpoints_path


def get_checkpoints_dir_path(experiment_name: str, ckpt_root_dir: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """Get the directory that includes all the checkpoints (and logs) of an experiment.
    ckpt_root_dir
        - experiment_name
            - run_id
                - ...
                - ...

    :param experiment_name:     Name of the experiment.
    :param ckpt_root_dir:       Path to the directory where all the experiments are organised, each sub-folder representing a specific experiment.
                                    If None, SG will first check if a package named 'checkpoints' exists.
                                    If not, SG will look for the root of the project that includes the script that was launched.
                                    If not found, raise an error.
    :param run_id:              Optional. Run id of the experiment. If None, the most recent run will be loaded.
    :return:                    Path of folder where the experiment checkpoints and logs will be stored.
    """
    experiment_dir = get_experiment_dir_path(checkpoints_root_dir=ckpt_root_dir, experiment_name=experiment_name)
    checkpoint_dir = experiment_dir if run_id is None else os.path.join(experiment_dir, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_experiment_dir_path(experiment_name: str, checkpoints_root_dir: Optional[str] = None) -> str:
    checkpoints_root_dir = checkpoints_root_dir or PKG_CHECKPOINTS_DIR or get_project_checkpoints_dir_path()
    if checkpoints_root_dir is None:
        raise ValueError("Illegal checkpoints directory: please set `ckpt_root_dir`")

    return os.path.join(checkpoints_root_dir, experiment_name)


def get_ckpt_local_path(experiment_name: str, ckpt_name: str, external_checkpoint_path: str, ckpt_root_dir: str = None, run_id: Optional[str] = None) -> str:
    """
    Gets the local path to the checkpoint file, which will be:
        - By default: YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name/ckpt_name.
        - external_checkpoint_path when external_checkpoint_path != None
        - ckpt_root_dir/experiment_name/ckpt_name when ckpt_root_dir != None.
        - if the checkpoint file is remotely located:
            when overwrite_local_checkpoint=True then it will be saved in a temporary path which will be returned,
            otherwise it will be downloaded to YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name and overwrite
            YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name/ckpt_name if such file exists.


    :param experiment_name:         Name of the experiment.
    :param ckpt_name:               Checkpoint filename
    :param external_checkpoint_path: Full path to checkpoint file (that might be located outside of super_gradients/checkpoints directory)
    :param ckpt_root_dir:           Path to the directory where all the experiments are organised, each sub-folder representing a specific experiment.
                                        If None, SG will first check if a package named 'checkpoints' exists.
                                        If not, SG will look for the root of the project that includes the script that was launched.
                                        If not found, raise an error.
    :param run_id:                  Optional. Run id of the experiment. If None, the most recent run will be loaded.
    :return:                        Path of folder where the experiment checkpoints and logs will be stored.
     :return: local path of the checkpoint file (Str)
    """
    if external_checkpoint_path:
        return external_checkpoint_path
    else:
        checkpoints_dir_path = get_checkpoints_dir_path(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name, run_id=run_id)
        return os.path.join(checkpoints_dir_path, ckpt_name)
