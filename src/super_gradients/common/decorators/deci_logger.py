def deci_func_logger(_func=None, *, name: str = "abstract_decorator"):
    """
    This decorator is used to wrap our functions with logs.
    It will log every enter and exit of the functon with the equivalent parameters as extras.
    It will also log exceptions that raises in the function.
    It will also log the exception time of the function.

    How it works:`
        First it will check if the decorator called with name keyword.
        If so it will return a new decorator that its logger is the name parameter.
        If not it will return a new decorator that its logger is the wrapped function name.
        Then the return decorator will return a new function that warps the original function with the new logs.
        For further understanding advise real-python "fancy decorators documentation"

    :param _func:    used when called without name specify. dont pass it directly
    :param name:     The name of the logger to save logs by.

    :return: a decorator that wraps function with logs logic.
    """
    # TODO: Not Working - Breaks the code, tests does not pass (s3 connector, platform...)
    # TODO: Fix problem with ExplicitParamValidation error (arguments not passed)
    # TODO: Run ALL test suite of deci2 (NOT circieCI test suite, but ALL the tests under tests folders)
    # TODO: Delete/Update all failing tests.

    # def deci_logger_decorator(fn):
    #
    # @functools.wraps(fn)
    # def wrapper_func(*args, **kwargs):
    #     try:
    #
    #         try:
    #             logger.debug(f"Start: {fn.__name__}", extra={"args": args, "kwargs": kwargs})
    #             time1 = time.perf_counter()
    #         except Exception:
    #             # failed to write log - continue.
    #             pass
    #
    #         result = fn(*args, **kwargs)
    #
    #         try:
    #             time2 = time.perf_counter()
    #             logger.debug(f"End: {fn.__name__}",
    #                          extra={'duration': (time2 - time1) * 1000.0, 'return_value': result})
    #         except Exception:
    #             # failed to write log - continue.
    #             pass
    #
    #         return result
    #
    #     except Exception as ex:
    #         # This exception was raised from inside the function call
    #         logger.error(f"Exception: {ex}", exc_info=ex)
    #         raise ex
    #
    # return wrapper_func

    # if _func is None:
    #     logger = get_logger(name)
    #     return deci_logger_decorator
    # else:
    #     logger = get_logger(_func.__name__)
    #     return deci_logger_decorator(_func)

    return _func


def deci_class_logger():
    """
    This decorator wraps every class method with deci_func_logger decorator.
    It works by checking if class method is callable and if so it will set a new decorated method as the same method name.
    """

    def wrapper(cls):
        # TODO: Not Working - Breaks the code, tests does not pass (s3 connector, platform...)
        # TODO: Fix problem with ExplicitParamValidation error (arguments not passed)
        # TODO: Run ALL test suite of deci2 (NOT circieCI test suite, but ALL the tests under tests folders)
        # TODO: Delete/Update all failing tests.

        # for attr in cls.__dict__:
        #     if callable(getattr(cls, attr)) and attr != '__init__':
        #         decorated_function = deci_func_logger(name=cls.__name__)(getattr(cls, attr))
        #         if type(cls.__dict__[attr]) is staticmethod:
        #             decorated_function = staticmethod(decorated_function)
        #         setattr(cls, attr, decorated_function)
        return cls

    return wrapper
