import os
import pkg_resources


try:
    PKG_CHECKPOINTS_DIR = pkg_resources.resource_filename("checkpoints", "")
except Exception:
    PKG_CHECKPOINTS_DIR = None


def get_checkpoints_dir(experiment_name: str, ckpt_root_dir: str = None) -> str:
    """Get the directory that includes all the checkpoints (and logs) of an experiment.

    :param experiment_name:     Name of the experiment.
    :param ckpt_root_dir:       Path to the directory where all the experiments are organised, each sub-folder including the checkpoints of a
                                specific experiment.
                                If None, it is assumed that pkg_resources.resource_filename('checkpoints', "") exists and will be used.
    :return:                    Path of folder where the experiment checkpoints and logs will be stored.
    """
    ckpt_root_dir = ckpt_root_dir or PKG_CHECKPOINTS_DIR
    if ckpt_root_dir is None:
        raise ValueError("Illegal checkpoints directory: please set ckpt_root_dir")
    return os.path.join(ckpt_root_dir, experiment_name)
