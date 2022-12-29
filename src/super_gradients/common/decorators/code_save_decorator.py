import inspect

import os

saved_codes = {}


def save_code(obj):
    """
    A decorator function which save the code of the class/function in a file (to be kept along with the training logs)
    File name will be according to the class/function name

    usage:

    @save_code
    MyClass():
        ...

        def foo():
            ...

        @save_code
        def do_something():
            ...

    This example will generate two files named MyClass.py and do_something.py, that will be saved in the checkpoint directory
    and uploaded to remote storage (if defined). the text of the class and the function will also be added to the tensorboard (or
    any other tracking service)
    """
    code = inspect.getsource(obj)
    name = obj.__name__
    saved_codes[name] = code
    return obj


def save_file(obj):
    """
    A decorator function which save the code of the entire file (to be kept along with the training logs).
    one call to this decorator in the file is enough to save the entire file


    usage:

    @save_file
    MyClass():
        ...

        def foo():
            ...

        def do_something():
            ...

    This example will save the file containing this code in the checkpoint directory and uploaded to remote storage (if defined).
    the content of the file will also be added to the tensorboard (or any other tracking service)

    """
    path = inspect.getsourcefile(obj)
    name = os.path.split(path)[-1]
    with open(path, "r") as f:
        code = f.read()
    saved_codes[name] = code
    return obj
