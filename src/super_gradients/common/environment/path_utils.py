def normalize_path(path: str) -> str:
    """Normalize the directory of file path. Replace the Windows-style (\\) path separators with unix ones (/).
    This is necessary when running on Windows since Hydra compose fails to find a configuration file is the config
    directory contains backward slash symbol.

    :param path: Input path string
    :return: Output path string with all \\ symbols replaces with /.
    """
    return path.replace("\\", "/")
