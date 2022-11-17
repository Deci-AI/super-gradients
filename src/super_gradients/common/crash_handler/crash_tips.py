import re
from typing import Union, Tuple, List, Type
from types import TracebackType

from super_gradients.common.crash_handler.utils import indent_string, fmt_txt, json_str_to_dict


class CrashTip:
    """Base class to add tips to exceptions raised while using SuperGradients.

    A tip is a more informative message with some suggestions for possible solutions or places to debug.
    """

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
            "- What went wrong:\n"
            f"{explanation}\n"
            "- How to solve it:\n"
            f"{solution}\n"
            f"- If the proposed solution did not help, feel free to contact the SuperGradient team or to open a ticket: "
            f"https://github.com/Deci-AI/super-gradients/issues/new/choose"
        )
        return msg

    @classmethod
    def get_message(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> Union[None, str]:
        """
        Wrap the tip in a nice message.

        Beside the class, the input params are as returned by sys.exc_info():
            :param cls:             Class inheriting from CrashTip
            :param exc_type:        Type of exception
            :param exc_value:       Exception
            :param exc_traceback:   Traceback

            :return:                Tip
        """
        try:
            tip = indent_string(cls._get_tip(exc_type, exc_value, exc_traceback), indent_size=4)
            message = (
                "******************************************************************************************************************************\n"
                "Something went wrong!\n"
                "Here is our guess:\n"
                f"{tip}\n"
                "see the trace above...\n"
                "******************************************************************************************************************************\n"
            )
            return "\n" + message
        except Exception:
            # It is important that the crash tip does not crash itself, because it is called atexit!
            # Otherwise, the user would get a crash on top of another crash and this would be extremly confusing
            return None


class TorchCudaMissingTip(CrashTip):
    @classmethod
    def is_relevant(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> bool:
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


class RecipeFactoryFormatTip(CrashTip):
    @classmethod
    def is_relevant(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> bool:
        pattern = "Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary"
        return isinstance(exc_value, RuntimeError) and pattern in str(exc_value)

    @classmethod
    def _get_explanation(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        factory_name, _ = RecipeFactoryFormatTip._get_factory_with_params(exc_value)
        factory_name = fmt_txt(factory_name, bold=True, color="green")
        msg = f"There is an indentation error in the recipe, while creating {factory_name}."
        return msg

    @classmethod
    def _get_solution(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        factory_name, params_dict = RecipeFactoryFormatTip._get_factory_with_params(exc_value)

        params_in_yaml = "\n".join(f"  {k}: {v}" for k, v in params_dict.items())
        user_yaml = f"- {factory_name}:\n" + params_in_yaml
        user_yaml = fmt_txt(user_yaml, indent=4, color="red")

        correct_yaml = f"- {factory_name}:\n" + indent_string(params_in_yaml, indent_size=2)
        correct_yaml = fmt_txt(correct_yaml, indent=4, color="green")
        msg = "If your yaml looks like this:\n" f"{user_yaml}\n" f"Then you change your recipe to this:\n" f"{correct_yaml}"

        return msg

    @staticmethod
    def _get_factory_with_params(exc_value: Exception) -> Tuple[str, dict]:
        """Utility function to extract useful features from the exception.
        :return: Name of the factory that (we assume) was not correctly defined
        :return: Parameters that are passed to that factory
        """
        description = str(exc_value)
        params_dict = re.search(r"received: (.*?)$", description).group(1)
        params_dict = json_str_to_dict(params_dict)
        factory_name = next(iter(params_dict))
        params_dict.pop(factory_name)
        return factory_name, params_dict


class DDPNotInitializedTip(CrashTip):
    """Note: I think that this should be caught within the code instead"""

    @classmethod
    def is_relevant(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType):
        expected_str = "Default process group has not been initialized, please make sure to call init_process_group."
        return isinstance(exc_value, RuntimeError) and expected_str in str(exc_value)

    @classmethod
    def _get_explanation(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        msg = "Your environment was not setup correctly for DDP."
        return msg

    @classmethod
    def _get_solution(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> str:
        msg = (
            "Please run at the beginning of your script:\n"
            f">>> {fmt_txt('from super_gradients.training.utils.distributed_training_utils import setup_gpu_mode', color='green')}\n"
            f">>> {fmt_txt('from super_gradients.common.data_types.enum import MultiGPUMode', color='green')}\n"
            f">>> {fmt_txt('setup_gpu_mode(gpu_mode=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=...)', color='green')}"
        )
        return msg


# /!\ Only the CrashTips classes listed below will be used !! /!\
ALL_CRASH_TIPS: List[Type[CrashTip]] = [TorchCudaMissingTip, RecipeFactoryFormatTip, DDPNotInitializedTip]


def get_relevant_crash_tip_message(exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> Union[None, str]:
    """Get a CrashTip class if relevant for input exception"""
    for crash_tip in ALL_CRASH_TIPS:
        if crash_tip.is_relevant(exc_type, exc_value, exc_traceback):
            return crash_tip.get_message(exc_type, exc_value, exc_traceback)
    return None
