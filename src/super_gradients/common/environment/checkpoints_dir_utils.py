import os
import pkg_resources


try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    PKG_CHECKPOINTS_DIR = None


def get_checkpoints_dir_path(experiment_name: str, ckpt_root_dir: str = None) -> str:
    """Get the directory that includes all the checkpoints (and logs) of an experiment.

    :param experiment_name:     Name of the experiment.
    :param ckpt_root_dir:       Path to the directory where all the experiments are organised, each sub-folder representing a specific experiment.
                                If None, it is assumed that pkg_resources.resource_filename('checkpoints', "") exists and will be used.
    :return:                    Path of folder where the experiment checkpoints and logs will be stored.
    """
    ckpt_root_dir = ckpt_root_dir or PKG_CHECKPOINTS_DIR
    if ckpt_root_dir is None:
        raise ValueError("Illegal checkpoints directory: please set ckpt_root_dir")
    return os.path.join(ckpt_root_dir, experiment_name)


def get_ckpt_local_path(source_ckpt_folder_name: str, experiment_name: str, ckpt_name: str, external_checkpoint_path: str):
    """
    Gets the local path to the checkpoint file, which will be:
        - By default: YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name.
        - if the checkpoint file is remotely located:
            when overwrite_local_checkpoint=True then it will be saved in a temporary path which will be returned,
            otherwise it will be downloaded to YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name and overwrite
            YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name/ckpt_name if such file exists.
        - external_checkpoint_path when external_checkpoint_path != None

    @param source_ckpt_folder_name: The folder where the checkpoint is saved. When set to None- uses the experiment_name.
    @param experiment_name: experiment name attr in trainer
    @param ckpt_name: checkpoint filename
    @param external_checkpoint_path: full path to checkpoint file (that might be located outside of super_gradients/checkpoints directory)
    @return:
    """
    if external_checkpoint_path:
        return external_checkpoint_path
    else:
        checkpoints_folder_name = source_ckpt_folder_name or experiment_name
        checkpoints_dir_path = get_checkpoints_dir_path(checkpoints_folder_name)
        return os.path.join(checkpoints_dir_path, ckpt_name)
