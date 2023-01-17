import sys
import unittest
import dataclasses
from typing import Type
import omegaconf
from super_gradients.common.crash_handler.crash_tips import (
    get_relevant_crash_tip_message,
    CrashTip,
    TorchCudaMissingTip,
    RecipeFactoryFormatTip,
    DDPNotInitializedTip,
    WrongHydraVersionTip,
    InterpolationKeyErrorTip,
)


@dataclasses.dataclass
class DocumentedException:
    exc_value: Exception
    expected_crash_tip: Type[CrashTip]
    # author/person who faced this exception?


class CrashTipTest(unittest.TestCase):
    def setUp(self) -> None:

        # Add any exception that we want to support here to make sure that it will be handled by our crash tip handler
        self.documented_exceptions = [
            DocumentedException(
                exc_value=OSError(
                    "/home/tomer.keren/.conda/envs/tomer-dev-sg3/lib/python3.10/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: symbol "
                    "cublasLtHSHMatmulAlgoInit version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference"
                ),
                expected_crash_tip=TorchCudaMissingTip,
            ),
            DocumentedException(
                exc_value=RuntimeError(
                    "Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary{type_name(str): "
                    "{parameters...}}.received: {'my_callback': None, 'lr_step': 2.4}"
                ),
                expected_crash_tip=RecipeFactoryFormatTip,
            ),
            DocumentedException(
                exc_value=RuntimeError("Default process group has not been initialized, please make sure to call init_process_group."),
                expected_crash_tip=DDPNotInitializedTip,
            ),
            DocumentedException(
                exc_value=TypeError("__init__() got an unexpected keyword argument 'version_base'"),
                expected_crash_tip=WrongHydraVersionTip,
            ),
            DocumentedException(
                exc_value=omegaconf.errors.InterpolationKeyError("omegaconf.errors.InterpolationKeyError: Interpolation key 'x' not found"),
                expected_crash_tip=InterpolationKeyErrorTip,
            ),
        ]

    def test_found_exceptions(self):
        """Test all the exceptions that were documented, and make sure that they have an associated tip."""
        for documented_exception in self.documented_exceptions:
            exc_value, expected_crash_tip = documented_exception.exc_value, documented_exception.expected_crash_tip
            try:
                raise exc_value
            except type(exc_value):
                exc_type, exc_value, exc_traceback = sys.exc_info()

                with self.subTest(
                    msg="Making sure that the CrashTip is considered relevant for the exception...",
                    expected_tip=expected_crash_tip.__name__,
                    exception=exc_value,
                ):
                    is_relevant = expected_crash_tip.is_relevant(exc_type, exc_value, exc_traceback)
                    self.assertTrue(
                        is_relevant,
                        msg=f"Crash tip '{expected_crash_tip.__name__}' should be relevant for exception '{exc_type.__name__}' but failed.",
                    )

                with self.subTest(
                    msg="Making sure that the CrashTip generates a message (None is returned if an error is raised internally, to avoid crashing atexit)...",
                    crash_tip=expected_crash_tip.__name__,
                ):
                    crash_tip_msg = expected_crash_tip.get_message(exc_type, exc_value, exc_traceback)
                    self.assertIsNotNone(
                        crash_tip_msg,
                        msg=f"The crash tip '{expected_crash_tip.__name__}' returned None, "
                        f"an exception was probably raised in '{expected_crash_tip.__name__}.get_message(...)'",
                    )

                with self.subTest(
                    msg="Making sure that we can find the relevant CrashTip and get it's summary for the exception...",
                    expected_tip=expected_crash_tip.__name__,
                    exception=exc_value,
                ):
                    crash_tip_message = get_relevant_crash_tip_message(exc_type, exc_value, exc_traceback)
                    expected_crash_tip_message = expected_crash_tip.get_message(exc_type, exc_value, exc_traceback)
                    self.assertEqual(
                        crash_tip_message,
                        expected_crash_tip_message,
                        msg=f"Crash tip message should be '{expected_crash_tip_message}' but got '{crash_tip_message}' instead.",
                    )
