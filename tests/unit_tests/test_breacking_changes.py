import unittest

from super_gradients.common.breaking_change import get_imports, compare_code


class TestBreakingChangeDetection(unittest.TestCase):
    def test_module_removed(self):
        old_code = "import package.missing_module"
        new_code = ""
        self.assertEqual(get_imports(old_code), {"package.missing_module": "package.missing_module"})
        self.assertEqual(get_imports(new_code), {})
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.imports_removed[0]
        self.assertEqual(breaking_change.import_name, "package.missing_module")

    def test_module_renamed(self):
        old_code = "import old_name"
        new_code = "import new_name"
        self.assertNotEqual(get_imports(old_code), get_imports(new_code))
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.imports_removed[0]
        self.assertEqual(breaking_change.import_name, "old_name")

    def test_module_location_changed(self):
        old_code = "from package import my_module"
        new_code = "from package.subpackage import my_module"
        self.assertNotEqual(get_imports(old_code), get_imports(new_code))
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.imports_removed[0]
        self.assertEqual(breaking_change.import_name, "package.my_module")

    def test_dependency_version_changed(self):
        """We want to be sensitive to source, not alias! (i.e. we want to distinguish between v1 and v2)"""

        old_code = "import library_v1 as library"
        new_code = "import library_v2 as library"
        self.assertNotEqual(get_imports(old_code), get_imports(new_code))
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.imports_removed[0]
        self.assertEqual(breaking_change.import_name, "library_v1")

    def test_function_removed(self):
        old_code = "def old_function(): pass"
        new_code = ""
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.functions_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.functions_removed[0]
        self.assertEqual(breaking_change.function_name, "old_function")

    def test_param_removed(self):
        old_code = "def my_function(param1, param2): pass"
        new_code = "def my_function(param1): pass"
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.params_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.params_removed[0]
        self.assertEqual(breaking_change.function_name, "my_function")

    def test_required_params_added(self):
        old_code = "def my_function(param1): pass"
        new_code = "def my_function(param1, param2): pass"
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.required_params_added), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.required_params_added[0]
        self.assertEqual(breaking_change.function_name, "my_function")
        self.assertEqual(breaking_change.parameter_name, "param2")

    def test_default_removed(self):
        old_code = "def my_function(param1=None): pass"
        new_code = "def my_function(param1): pass"
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.required_params_added), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.required_params_added[0]
        self.assertEqual(breaking_change.function_name, "my_function")
        self.assertEqual(breaking_change.parameter_name, "param1")

    def test_optional_param_added(self):
        old_code = "def my_function(param1): pass"
        new_code = "def my_function(param1, param2=None): pass"
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.required_params_added), 0)

    def test_optional_param_added2(self):
        old_code = "def my_function(param=None): pass"
        new_code = "def my_function(): pass"
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.params_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.params_removed[0]
        self.assertEqual(breaking_change.function_name, "my_function")
        self.assertEqual(breaking_change.parameter_name, "param")

    def test_no_changes(self):
        old_code = "def my_function(param1): pass"
        new_code = "def my_function(param1): pass"
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.functions_removed), 0)
        self.assertEqual(len(breaking_changes.params_removed), 0)
        self.assertEqual(len(breaking_changes.required_params_added), 0)
        self.assertEqual(len(breaking_changes.imports_removed), 0)

    def test_multiple_changes(self):
        old_code = "import module1\nfrom module2 import function1\ndef my_function(param1, param2): pass"
        new_code = "import module3\ndef my_function(param1, param2, param3): pass"
        breaking_changes = compare_code("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 2)
        self.assertEqual(len(breaking_changes.required_params_added), 1)


if __name__ == "__main__":
    unittest.main()
