import inspect
import functools
from typing import Callable


class _ExplicitParamsValidator:
    def __init__(self, function: Callable, validation_type: str = 'None'):
        """
        ExplicitParamsValidator
            :param function:
            :param validation_type:
        """
        self.func = function
        self.validation_type = validation_type

    def __get__(self, obj, type=None):
        return functools.partial(self, obj)

    def __call__(self, *args, **kwargs):
        """
        Caller to create the Wrapper for
            :param args:
            :param kwargs:
            :return:
        """
        if not hasattr(self, 'func'):
            self.func = args[0]
            return self

        return self.validate_explicit_params(*args, **kwargs)

    def validate_explicit_params(self, *args, **kwargs):
        """
        validate_explicit_params - Checks if each of the explicit parameters (The ones the user put in the func/method
                                   call) are not None or Empty - Depending on the instantiation
            :param args:  The function arguments
            :param kwargs: The function Keyword arguments
            :return: wrapped function after input params None values validation
        """
        var_names = inspect.getfullargspec(self.func)[0]

        explicit_args_var_names = list(var_names[:len(args)])

        # FOR CLASS METHOD REMOVE THE EXPLICIT DEMAND FOR self PARAMETER
        for params_list in [explicit_args_var_names, list(kwargs.keys())]:
            if 'self' in params_list:
                params_list.remove('self')

        # FIRST OF ALL HANDLE ALL OF THE KEYWORD ARGUMENTS
        for kwarg, value in kwargs.items():
            self.__validate_input_param(kwarg, value)

        for i, input_param in enumerate(explicit_args_var_names):
            self.__validate_input_param(input_param, args[i])

        return self.func(*args, **kwargs)

    def __validate_input_param(self, input_param, value):
        """
        __validate_input_param - Validates the input param based on the validation type
                                 in the class constructor
        :param input_param:
        :param value:
        :return:
        """
        if self.validation_type == 'NoneOrEmpty':
            if not value:
                raise ValueError('Input param: ' + str(input_param) + ' is Empty')

        if value is None:
            raise ValueError('Input param: ' + str(input_param) + ' is None')


# WRAPS THE RETRY DECORATOR CLASS TO ENABLE CALLING WITHOUT PARAMS
def explicit_params_validation(function: Callable = None, validation_type: str = 'None'):
    if function is not None:
        return _ExplicitParamsValidator(function=function)
    else:
        def wrapper(function):
            return _ExplicitParamsValidator(function=function, validation_type=validation_type)

        return wrapper
