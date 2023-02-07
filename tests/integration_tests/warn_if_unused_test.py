import io
import unittest

from super_gradients.training import models


class WarnIfUnusedIntegrationTest(unittest.TestCase):
    """Check that warn_if_unused is properly integrated into different SuperGradients components."""

    def test_unused_arch_params(self):
        from contextlib import redirect_stderr

        stderr_buffer = io.StringIO()
        with redirect_stderr(stderr_buffer):
            _ = models.get("resnet18", arch_params={"num_classes": 123, "dummy_val": None, "dummy_list": [0, 1, 2, 3]})

        self.assertIn(member="contains parameters that are not required and will be ignored", container=stderr_buffer.getvalue())

    def test_used_arch_params(self):
        from contextlib import redirect_stderr

        stderr_buffer = io.StringIO()
        with redirect_stderr(stderr_buffer):
            _ = models.get("resnet18", arch_params={"num_classes": 123})

        self.assertNotIn(member="contains parameters that are not required and will be ignored", container=stderr_buffer.getvalue())
