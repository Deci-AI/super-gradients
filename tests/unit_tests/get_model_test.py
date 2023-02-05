import io
import unittest

from super_gradients.training import models
from super_gradients.training.utils.config_utils import IS_UNUSED_MESSAGE_INTRO


class GetModelTest(unittest.TestCase):
    def test_unused_arch_params(self):
        from contextlib import redirect_stderr

        stderr_buffer = io.StringIO()
        with redirect_stderr(stderr_buffer):
            _ = models.get("resnet18", arch_params={"num_classes": 123, "dummy_val": None, "dummy_list": [0, 1, 2, 3]})

        self.assertIn(member=IS_UNUSED_MESSAGE_INTRO, container=stderr_buffer.getvalue())

    def test_only_used_arch_params(self):
        from contextlib import redirect_stderr

        stderr_buffer = io.StringIO()
        with redirect_stderr(stderr_buffer):
            _ = models.get("resnet18", arch_params={"num_classes": 123})

        self.assertNotIn(member=IS_UNUSED_MESSAGE_INTRO, container=stderr_buffer.getvalue())
