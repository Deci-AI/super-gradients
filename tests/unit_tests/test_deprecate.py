import unittest
import warnings
from super_gradients.common.deprecate import deprecate_call


class TestDeprecationDecorator(unittest.TestCase):
    def setUp(self):
        """Prepare required functions before each test."""
        self.new_function_message = "This is the new function!"

        def new_func():
            return self.new_function_message

        @deprecate_call(deprecated_in_v="3.2.0", removed_in_v="4.0.0", target=new_func, reason="Replaced for optimization")
        def fully_configured_deprecated_func():
            return new_func()

        @deprecate_call(
            deprecated_in_v="3.2.0",
            removed_in_v="4.0.0",
        )
        def basic_deprecated_func():
            return new_func()

        self.new_func = new_func
        self.fully_configured_deprecated_func = fully_configured_deprecated_func
        self.basic_deprecated_func = basic_deprecated_func

    def test_emits_warning(self):
        """Ensure that the deprecated function emits a warning when called."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.fully_configured_deprecated_func()
            self.assertEqual(len(w), 1)

    def test_displays_deprecated_version(self):
        """Ensure that the warning contains the version in which the function was deprecated."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.fully_configured_deprecated_func()
            self.assertTrue(any("3.2.0" in str(warning.message) for warning in w))

    def test_displays_removed_version(self):
        """Ensure that the warning contains the version in which the function will be removed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.fully_configured_deprecated_func()
            self.assertTrue(any("4.0.0" in str(warning.message) for warning in w))

    def test_guidance_on_replacement(self):
        """Ensure that if a replacement target is provided, guidance on using the new function is included in the warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.fully_configured_deprecated_func()
            self.assertTrue(any("new_func" in str(warning.message) for warning in w))

    def test_displays_reason(self):
        """Ensure that if provided, the reason for deprecation is included in the warning."""
        reason_str = "Replaced for optimization"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.fully_configured_deprecated_func()
            self.assertTrue(any(reason_str in str(warning.message) for warning in w))

    def test_triggered_only_once(self):
        """Ensure that the deprecation warning is triggered only once even if the deprecated function is called multiple times."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(10):
                self.fully_configured_deprecated_func()
            self.assertEqual(len(w), 1, "Only one warning should be emitted")

    def test_basic_deprecation_emits_warning(self):
        """Ensure that a function with minimal deprecation configuration emits a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.basic_deprecated_func()
            self.assertEqual(len(w), 1)


if __name__ == "__main__":
    unittest.main()
