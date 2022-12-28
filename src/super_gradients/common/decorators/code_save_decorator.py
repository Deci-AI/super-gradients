import inspect

import os

saved_codes = {}


def save_code(obj):
    """
    A decorator function which save the code of the class/function in a file (to be kept along with the training logs)
    File name will be according to the class/function name
    """
    code = inspect.getsource(obj)
    name = obj.__name__
    saved_codes[name] = code
    return obj


def save_file(obj):
    """
    A decorator function which save the code of the entire file (to be kept along with the training logs).
    one call to this decorator in the file is enough to save the entire file
    """
    path = inspect.getsourcefile(obj)
    name = os.path.split(path)[-1]
    with open(path, "r") as f:
        code = f.read()
    saved_codes[name] = code
    return obj
