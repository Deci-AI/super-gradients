from typing import Union
from types import TracebackType

from super_gradients.common.crash_handler.crash_tips import get_relevant_crash_tip


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
        crash_tip = get_relevant_crash_tip(ExceptionInfo.exc_type, ExceptionInfo.exc_value, ExceptionInfo.exc_traceback)
        if crash_tip:
            return crash_tip.get_message(ExceptionInfo.exc_type, ExceptionInfo.exc_value, ExceptionInfo.exc_traceback)
