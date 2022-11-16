import sys
import unittest
import dataclasses

from super_gradients.common.crash_handler import get_relevant_crash_tip
from super_gradients.common.crash_handler.crash_tips import TorchCudaMissing, RecipeFactoryFormat, DDPNotInitialized


@dataclasses.dataclass
class EncounteredException:
    expected_crash_tip: object
    exc_value: Exception
    # author/person who faced this exception?


class CrashTipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encountered_exceptions = [
            EncounteredException(
                exc_value=OSError(
                    "/home/tomer.keren/.conda/envs/tomer-dev-sg3/lib/python3.10/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: symbol "
                    "cublasLtHSHMatmulAlgoInit version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference"
                ),
                expected_crash_tip=TorchCudaMissing,
            ),
            EncounteredException(
                exc_value=RuntimeError(
                    "Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary{type_name(str): "
                    "{parameters...}}.received: {'my_callback': None, 'lr_step': 2.4}"
                ),
                expected_crash_tip=RecipeFactoryFormat,
            ),
            EncounteredException(
                exc_value=RuntimeError("Default process group has not been initialized, please make sure to call init_process_group."),
                expected_crash_tip=DDPNotInitialized,
            ),
        ]

    def test_found_exceptions(self):
        """Test all the exceptions that were documented, and make sure that they have an associated tip."""
        for encountered_exception in self.encountered_exceptions:
            exc_value, expected_crash_tip = encountered_exception.exc_value, encountered_exception.expected_crash_tip
            try:
                raise exc_value
            except type(exc_value):
                exc_type, exc_value, exc_traceback = sys.exc_info()

                with self.subTest(msg="testing get_relevant_crash_tip", expected_tip=expected_crash_tip, exception=exc_value):
                    crash_tip = get_relevant_crash_tip(exc_type, exc_value, exc_traceback)
                    self.assertEqual(
                        crash_tip,
                        expected_crash_tip,
                        msg=f"Crash tip {expected_crash_tip} was expected but got {crash_tip} instead. ",
                    )

                with self.subTest(msg="testing get_message", crash_tip=expected_crash_tip):
                    crash_tip_msg = expected_crash_tip.get_message(exc_type, exc_value, exc_traceback)
                    self.assertIsNotNone(
                        crash_tip_msg,
                        msg=f"The crash tip {expected_crash_tip} returned None, an exception was probably raised within {crash_tip}.get_message",
                    )
