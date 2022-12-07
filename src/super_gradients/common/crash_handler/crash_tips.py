import re
from typing import Union, Tuple, List, Type
from types import TracebackType

from super_gradients.common.crash_handler.utils import indent_string, fmt_txt, json_str_to_dict
from super_gradients.common.abstractions.abstract_logger import get_logger


logger = get_logger(__name__)


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
    def _get_tips(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> List[str]:
        """
        Provide a customized tip for the exception, combining explanation and solution.

        Beside the class, the input params are as returned by sys.exc_info():
            :param cls:             Class inheriting from CrashTip
            :param exc_type:        Type of exception
            :param exc_value:       Exception
            :param exc_traceback:   Traceback

            :return:                Tip
        """
        raise NotImplementedError

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

            def format_tip(tip_index: int, tip: str):
                first_sentence, *following_sentences = tip.split("\n")
                first_sentence = f"{tip_index+1}. {first_sentence}"
                following_sentences = [f"   {sentence}" for sentence in following_sentences]
                return "\n".join([first_sentence] + following_sentences)

            tips: List[str] = cls._get_tips(exc_type, exc_value, exc_traceback)
            formatted_tips: str = "\n".join([format_tip(i, tip) for i, tip in enumerate(tips)])

            message = (
                "═══════════════════════════════════════════╦═════════════════════════╦════════════════════════════════════════════════════════════\n"
                "                                           ║ SuperGradient Crash tip ║ \n"
                "                                           ╚═════════════════════════╝ \n"
                f"{fmt_txt('Something went wrong!', color='red', bold=True)} You can find below potential solution(s) to this error: \n\n"
                f"{formatted_tips}\n"
                f"{len(tips)+1}. If the proposed solution(s) did not help, feel free to contact the SuperGradient team or to open a ticket on "
                f"https://github.com/Deci-AI/super-gradients/issues/new/choose\n\n"
                "see the trace above...\n"
                "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n"
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
    def _get_tips(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> List[str]:
        tip = (
            f"This error may indicate {fmt_txt('CUDA libraries version conflict', color='red')} (When Torchvision & Torch are installed for different "
            f"CUDA versions) or the {fmt_txt('absence of CUDA support in PyTorch', color='red')}.\n"
            "To fix this you can:\n"
            f"   a. Make sure to {fmt_txt('uninstall torch, torchvision', color='green')}\n"
            f"   b. {fmt_txt('Install the torch version', color='green')} that respects your os & compute platform "
            f"{fmt_txt('following the instruction from https://pytorch.org/', color='green')}"
        )
        return [tip]


class RecipeFactoryFormatTip(CrashTip):
    @classmethod
    def is_relevant(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> bool:
        pattern = "Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary"
        return isinstance(exc_value, RuntimeError) and pattern in str(exc_value)

    @classmethod
    def _get_tips(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> List[str]:
        factory_name, params_dict = RecipeFactoryFormatTip._get_factory_with_params(exc_value)

        formatted_factory_name = fmt_txt(factory_name, bold=True, color="green")

        params_in_yaml = "\n".join(f"  {k}: {v}" for k, v in params_dict.items())
        user_yaml = f"- {factory_name}:\n" + params_in_yaml
        formatted_user_yaml = fmt_txt(user_yaml, indent=4, color="red")

        correct_yaml = f"- {factory_name}:\n" + indent_string(params_in_yaml, indent_size=2)
        formatted_correct_yaml = fmt_txt(correct_yaml, indent=4, color="green")

        tip = f"There is an indentation error in the recipe, while creating {formatted_factory_name}.\n"
        tip += "If your wrote this in your recipe:\n"
        tip += f"{formatted_user_yaml}\n"
        tip += "Please change it to:\n"
        tip += f"{formatted_correct_yaml}"
        tips = [tip]
        return tips

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
    def _get_tips(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> List[str]:
        tip = (
            "Your environment was not setup correctly for DDP.\n"
            "Please run at the beginning of your script:\n"
            f">>> {fmt_txt('from super_gradients.training.utils.distributed_training_utils import setup_device', color='green')}\n"
            f">>> {fmt_txt('from super_gradients.common.data_types.enum import MultiGPUMode', color='green')}\n"
            f">>> {fmt_txt('setup_device(multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=...)', color='green')}"
        )
        return [tip]


class WrongHydraVersionTip(CrashTip):
    """Note: I think that this should be caught within the code instead"""

    @classmethod
    def is_relevant(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType):
        expected_str = "__init__() got an unexpected keyword argument 'version_base'"
        return isinstance(exc_value, TypeError) and expected_str == str(exc_value)

    @classmethod
    def _get_tips(cls, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> List[str]:
        import hydra

        tip = (
            f"{fmt_txt(f'hydra=={hydra.__version__}', color='red')} is not supported by SuperGradients. "
            f"Please run {fmt_txt('pip install hydra-core==1.2.0', color='green')}"
        )
        return [tip]


# /!\ Only the CrashTips classes listed below will be used !! /!\
ALL_CRASH_TIPS: List[Type[CrashTip]] = [TorchCudaMissingTip, RecipeFactoryFormatTip, DDPNotInitializedTip, WrongHydraVersionTip]


def get_relevant_crash_tip_message(exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> Union[None, str]:
    """Get a CrashTip class if relevant for input exception"""
    for crash_tip in ALL_CRASH_TIPS:
        if crash_tip.is_relevant(exc_type, exc_value, exc_traceback):
            return crash_tip.get_message(exc_type, exc_value, exc_traceback)
    return None
