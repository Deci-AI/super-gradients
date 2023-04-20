from super_gradients.common.environment.argparse_utils import pop_arg


def pop_local_rank() -> int:
    """Pop the python arg "local-rank". If exists inform the user with a log, otherwise return -1."""
    try:
        local_rank = int(pop_arg("local_rank", default_value=-1))
    except ValueError:
        local_rank = -1

    return local_rank
