import sys
from types import TracebackType


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
    def get_exception():
        return ExceptionInfo.exc_type, ExceptionInfo.exc_value, ExceptionInfo.exc_traceback


def register_exceptions(excepthook):
    """Wrap excepthook with a step the saves the exception info to be available in the exit hooks."""

    def excepthook_with_register(exc_type, exc_value, exc_traceback):
        ExceptionInfo.register_exception(exc_type, exc_value, exc_traceback)
        return excepthook(exc_type, exc_value, exc_traceback)

    return excepthook_with_register


sys.excepthook = register_exceptions(sys.excepthook)
