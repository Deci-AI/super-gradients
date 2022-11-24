from typing import Union, Callable
from types import TracebackType

from super_gradients.common.crash_handler.crash_tips import get_relevant_crash_tip_message


def register_exceptions(excepthook: Callable) -> Callable:
    """Wrap excepthook with a step the saves the exception info to be available in the exit hooks.
    :param exc_type:        Type of exception
    :param exc_value:       Exception
    :param exc_traceback:   Traceback

    :return: wrapped exceptook, that register the exception before raising it
    """

    def excepthook_with_register(exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> Callable:
        ExceptionInfo.register_exception(exc_type, exc_value, exc_traceback)
        return excepthook(exc_type, exc_value, exc_traceback)

    return excepthook_with_register


class ExceptionInfo:
    """Holds information about the session exception (if any)"""

    _is_exception_raised = False
    exc_type = None
    exc_value = None
    exc_traceback = None

    @staticmethod
    def register_exception(exc_type: type, exc_value: Exception, exc_traceback: TracebackType):
        """Register the exception information into the class"""
        ExceptionInfo._is_exception_raised = True
        ExceptionInfo.exc_type = exc_type
        ExceptionInfo.exc_value = exc_value
        ExceptionInfo.exc_traceback = exc_traceback

    @staticmethod
    def is_exception_raised():
        """Check if an exception was raised in the current process"""
        return ExceptionInfo._is_exception_raised

    @staticmethod
    def get_crash_tip_message() -> Union[None, str]:
        return get_relevant_crash_tip_message(ExceptionInfo.exc_type, ExceptionInfo.exc_value, ExceptionInfo.exc_traceback)
