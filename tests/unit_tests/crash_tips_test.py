import dataclasses
import sys
import unittest
from super_gradients.common.crash_handler import get_relevant_crash_tip


@dataclasses.dataclass
class EncounteredException:
    name: str
    exc_value: Exception


class DataLoaderFactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.exceptions = [
            EncounteredException(
                name="torch_cuda_version_error",
                exc_value=OSError(
                    "/home/tomer.keren/.conda/envs/tomer-dev-sg3/lib/python3.10/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: symbol "
                    "cublasLtHSHMatmulAlgoInit version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference"
                ),
            ),
            EncounteredException(
                name="hydra_yaml_format",
                exc_value=RuntimeError(
                    "Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary{type_name(str): "
                    "{parameters...}}.received: {'my_callback': None, 'lr_step': 2.4}"
                ),
            ),
        ]

    def test_existing_exceptions(self):
        for encountered_exc in self.exceptions:
            with self.subTest(msg=f"testing exception: {encountered_exc.name}", exception=encountered_exc.exc_value):
                try:
                    raise encountered_exc.exc_value
                except type(encountered_exc.exc_value):
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    crash_tip = get_relevant_crash_tip(exc_type, exc_value, exc_traceback)

                    self.assertIsNotNone(crash_tip, msg=f"No crash tip found for the exception '{encountered_exc.name}': '{encountered_exc.exc_value}'")

                    crash_tip_msg = crash_tip.get_message(exc_type, exc_value, exc_traceback)
                    self.assertIsNotNone(
                        crash_tip_msg, msg=f"The crash tip {crash_tip} returned None, an exception was probably raised within {crash_tip}.get_message"
                    )
