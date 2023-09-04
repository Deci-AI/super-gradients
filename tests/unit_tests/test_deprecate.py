import warnings
import unittest
from unittest.mock import patch

from super_gradients.common.deprecate import deprecated


class TestDeprecationDecorator(unittest.TestCase):
    def setUp(self):
        """Prepare required functions before each test."""
        self.new_function_message = "This is the new function!"

        def new_func():
            return self.new_function_message

        @deprecated(deprecated_since="3.2.0", removed_from="10.0.0", target=new_func, reason="Replaced for optimization")
        def fully_configured_deprecated_func():
            return new_func()

        @deprecated(deprecated_since="3.2.0", removed_from="10.0.0")
        def basic_deprecated_func():
            return new_func()

        self.new_func = new_func
        self.fully_configured_deprecated_func = fully_configured_deprecated_func
        self.basic_deprecated_func = basic_deprecated_func

        class NewClass:
            def __init__(self):
                pass

        @deprecated(deprecated_since="3.2.0", removed_from="10.0.0", target=NewClass, reason="Replaced for optimization")
        class DeprecatedClass:
            def __init__(self):
                pass

        self.NewClass = NewClass
        self.DeprecatedClass = DeprecatedClass

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
            self.assertTrue(any("10.0.0" in str(warning.message) for warning in w))

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

    def test_class_deprecation_warning(self):
        """Ensure that creating an instance of a deprecated class emits a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = self.DeprecatedClass()  # Instantiate the deprecated class
            self.assertEqual(len(w), 1)

    def test_class_deprecation_message_content(self):
        """Ensure that the emitted warning for a deprecated class contains relevant information including target class."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = self.DeprecatedClass()
            self.assertTrue(any("3.2.0" in str(warning.message) for warning in w))
            self.assertTrue(any("10.0.0" in str(warning.message) for warning in w))
            self.assertTrue(any("DeprecatedClass" in str(warning.message) for warning in w))
            self.assertTrue(any("Replaced for optimization" in str(warning.message) for warning in w))
            self.assertTrue(any("NewClass" in str(warning.message) for warning in w))

    def test_raise_error_when_library_version_equals_removal_version(self):
        """Ensure that an error is raised when the library's version equals the function's removal version."""
        with patch("super_gradients.__version__", "10.1.0"):  # Mocking the version to be equal to removal version
            with self.assertRaises(ImportError):

                @deprecated(deprecated_since="3.2.0", removed_from="10.1.0", target=self.new_func)
                def deprecated_func_version_equal():
                    return

                deprecated_func_version_equal()

    def test_no_error_when_library_version_below_removal_version(self):
        """Ensure that no error is raised when the library's version is below the function's removal version."""
        with patch("super_gradients.__version__", "10.1.0"):  # Mocking the version to be below removal version

            @deprecated(deprecated_since="3.2.0", removed_from="10.2.0", target=self.new_func)
            def deprecated_func_version_below():
                return

            deprecated_func_version_below()


if __name__ == "__main__":
    unittest.main()
