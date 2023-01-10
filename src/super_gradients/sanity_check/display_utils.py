def format_error_msg(test_name: str, error_msg: str) -> str:
    """Format an error message in the appropriate format.

    :param test_name:   Name of the test being tested.
    :param error_msg:   Message to format in appropriate format.
    :return:            Formatted message
    """
    return f"\33[31mFailed to verify {test_name}: {error_msg}\33[0m"
