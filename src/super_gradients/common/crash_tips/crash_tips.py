import atexit
import dataclasses

from types import TracebackType
from ..crash_tips.exception import ExceptionInfo


def indent_string(txt: str, indent_size: int) -> str:
    """Add an indentation to a string."""
    indent = " " * indent_size
    return indent + txt.replace("\n", "\n" + indent)


class CrashTip:
    """Base class to add tips to exceptions raised while using SuperGradients.

    A tip is a more informative message with some suggestions for possible solutions or places to debug.

    Example:
        >>> try:
        >>>     raise RuntimeError("dummy_exception")
        >>> except:
        >>>     exc_type, exc_value, exc_traceback = sys.exc_info()
        >>>     if MyTip.is_relevant(exc_type, exc_value, exc_traceback):
        >>>         MyTip.get_message(exc_type, exc_value, exc_traceback)
    """

    # https://docs.python.org/3/library/traceback.html#traceback.StackSummary

    @classmethod
    def is_relevant(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> bool:
        """
        Check if this tip is relevant.

        Beside the class, the input params are as returned by sys.exc_info():
            :param cls:             Class inheriting from CrashTip
            :param exc_type:        Type of exception
            :param exc_value:       Exception
            :param exc_traceback:   Traceback

            :return:                True if the current class can help with the exception
        """
        raise NotImplementedError

    @classmethod
    def _get_explanation(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        """
        Propose an explanation on what caused the exception to be raised.

        Beside the class, the input params are as returned by sys.exc_info():
            :param cls:             Class inheriting from CrashTip
            :param exc_type:        Type of exception
            :param exc_value:       Exception
            :param exc_traceback:   Traceback

            :return:                Explanation
        """
        raise NotImplementedError

    @classmethod
    def _get_solution(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        """
        Propose a solution (or more) to avoid the exception

        Beside the class, the input params are as returned by sys.exc_info():
            :param cls:             Class inheriting from CrashTip
            :param exc_type:        Type of exception
            :param exc_value:       Exception
            :param exc_traceback:   Traceback

            :return:                Solution
        """
        raise NotImplementedError

    @classmethod
    def _get_tip(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        """
        Provide a customized tip for the exception, combining explanation and solution.

        Beside the class, the input params are as returned by sys.exc_info():
            :param cls:             Class inheriting from CrashTip
            :param exc_type:        Type of exception
            :param exc_value:       Exception
            :param exc_traceback:   Traceback

            :return:                Tip
        """
        explanation = indent_string(cls._get_explanation(exc_type, exc_value, exc_traceback), indent_size=4)
        solution = indent_string(cls._get_solution(exc_type, exc_value, exc_traceback), indent_size=4)

        msg = (
            "- What we think happened:\n"
            f"{explanation}\n"
            "- What we think could solve it:\n"
            f"{solution}\n"
            f"- If the proposed solution did not help, feel free to contact us or to open a ticket: "
            f"https://github.com/Deci-AI/super-gradients/issues/new/choose"
        )
        return msg

    @classmethod
    def get_message(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        """
        Wrap the tip in a nice message.

        Beside the class, the input params are as returned by sys.exc_info():
            :param cls:             Class inheriting from CrashTip
            :param exc_type:        Type of exception
            :param exc_value:       Exception
            :param exc_traceback:   Traceback

            :return:                Tip
        """
        tip = indent_string(cls._get_tip(exc_type, exc_value, exc_traceback), indent_size=4)
        message = (
            "******************************************************************************************************************************\n"
            "Something went wrong!\n"
            f"{tip}\n"
            "see the trace above...\n"
            "******************************************************************************************************************************\n"
        )
        return "\n" + message


class TorchCudaMissing(CrashTip):
    @classmethod
    def is_relevant(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType):
        pattern = "symbol cublasLtHSHMatmulAlgoInit version"
        return isinstance(exc_value, OSError) and pattern in str(exc_value)

    @classmethod
    def _get_explanation(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        msg = "The torch version installed is not compatible with your CUDA version. "
        return msg

    @classmethod
    def _get_solution(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        msg = (
            "1. Make sure to uninstall torch, torchvision and torchaudio\n"
            "2. Install the torch version that respects your os & compute platform following the instruction from https://pytorch.org/"
        )
        return msg


ALL_CRASH_TIPS = [TorchCudaMissing]


def crash_tip_handler():
    """Display a crash tip if an error was raised"""
    if ExceptionInfo.is_exception_raised():
        exc_type, exc_value, exc_traceback = ExceptionInfo.get_exception()
        for crash_tip in ALL_CRASH_TIPS:
            if crash_tip.is_relevant(exc_type, exc_value, exc_traceback):
                print(crash_tip.get_message(exc_type, exc_value, exc_traceback))


atexit.register(crash_tip_handler)


@dataclasses.dataclass
class EncounteredException:
    exc_value: Exception
    author: str


tomer_torch_error = EncounteredException(
    author="tomer",
    exc_value=OSError(
        "/home/tomer.keren/.conda/envs/tomer-dev-sg3/lib/python3.10/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: symbol "
        "cublasLtHSHMatmulAlgoInit version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference"
    ),
)
tomer_yaml_error = EncounteredException(
    author="tomer",
    exc_value=RuntimeError(
        "Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary{type_name(str): "
        "{parameters...}}.received: {'my_callback': None, 'lr_step': 2.4}'}"
    ),
)

not_tomer_error = OSError(
    "/home/tomer.keren/.conda/envs/tomer-dev-sg3/lib/python3.10/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: something else"
)

raise tomer_torch_error.exc_value
