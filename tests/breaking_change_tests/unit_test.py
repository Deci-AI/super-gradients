"""
This unittest is NOT added to the unit_tests folder because it is NOT testing SG code.
The breaking change tests are only meant to exist in the CI, and therefore it was not included to the SG package.
"""
import unittest

from .code_parser import FunctionParameter, FunctionParameters, FunctionSignature, parse_imports, parse_functions_signatures
from .breaking_changes_detection import extract_code_breaking_changes


class TestBreakingChangeDetection(unittest.TestCase):
    def test_module_removed(self):
        old_code = "import super_gradients.missing_module"
        new_code = ""
        self.assertEqual(parse_imports(old_code), {"super_gradients.missing_module": "super_gradients.missing_module"})
        self.assertEqual(parse_imports(new_code), {})

        # Imports not checked in regular modules
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 0)

        # Imports checked in  __init__.py
        breaking_changes = extract_code_breaking_changes("__init__.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 1)

        # Attributes of the breaking change
        breaking_change = breaking_changes.imports_removed[0]
        self.assertEqual(breaking_change.import_name, "super_gradients.missing_module")

    def test_module_renamed(self):
        old_code = "import super_gradients"
        new_code = "import new_module"
        self.assertNotEqual(parse_imports(old_code), parse_imports(new_code))

        # Imports not checked in regular modules
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 0)

        # Imports checked in  __init__.py
        breaking_changes = extract_code_breaking_changes("__init__.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.imports_removed[0]
        self.assertEqual(breaking_change.import_name, "super_gradients")

    def test_module_location_changed(self):
        old_code = "from super_gradients import my_module"
        new_code = "from super_gradients.subpackage import my_module"
        self.assertNotEqual(parse_imports(old_code), parse_imports(new_code))

        # Imports not checked in regular modules
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 0)

        # Imports checked in  __init__.py
        breaking_changes = extract_code_breaking_changes("__init__.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.imports_removed[0]
        self.assertEqual(breaking_change.import_name, "super_gradients.my_module")

    def test_dependency_version_changed(self):
        """We want to be sensitive to source, not alias! (i.e. we want to distinguish between v1 and v2)"""

        old_code = "import super_gradients.library_v1 as library"
        new_code = "import super_gradients.library_v2 as library"
        self.assertNotEqual(parse_imports(old_code), parse_imports(new_code))

        # Imports not checked in regular modules
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 0)

        # Imports checked in  __init__.py
        breaking_changes = extract_code_breaking_changes("__init__.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.imports_removed[0]
        self.assertEqual(breaking_change.import_name, "super_gradients.library_v1")

    def test_function_removed(self):
        old_code = "def old_function(): pass"
        new_code = ""
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.functions_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.functions_removed[0]
        self.assertEqual(breaking_change.function_name, "old_function")

    def test_param_removed(self):
        old_code = "def my_function(param1, param2): pass"
        new_code = "def my_function(param1): pass"
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.params_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.params_removed[0]
        self.assertEqual(breaking_change.function_name, "my_function")

    def test_required_params_added(self):
        old_code = "def my_function(param1): pass"
        new_code = "def my_function(param1, param2): pass"
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.required_params_added), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.required_params_added[0]
        self.assertEqual(breaking_change.function_name, "my_function")
        self.assertEqual(breaking_change.parameter_name, "param2")

    def test_default_removed(self):
        old_code = "def my_function(param1=None): pass"
        new_code = "def my_function(param1): pass"
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.required_params_added), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.required_params_added[0]
        self.assertEqual(breaking_change.function_name, "my_function")
        self.assertEqual(breaking_change.parameter_name, "param1")

    def test_optional_param_added(self):
        old_code = "def my_function(param1): pass"
        new_code = "def my_function(param1, param2=None): pass"
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.required_params_added), 0)

    def test_optional_param_added2(self):
        old_code = "def my_function(param=None): pass"
        new_code = "def my_function(): pass"
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.params_removed), 1)

        # Check the attributes of the breaking change
        breaking_change = breaking_changes.params_removed[0]
        self.assertEqual(breaking_change.function_name, "my_function")
        self.assertEqual(breaking_change.parameter_name, "param")

    def test_no_changes(self):
        old_code = "def my_function(param1): pass"
        new_code = "def my_function(param1): pass"
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.functions_removed), 0)
        self.assertEqual(len(breaking_changes.params_removed), 0)
        self.assertEqual(len(breaking_changes.required_params_added), 0)
        self.assertEqual(len(breaking_changes.imports_removed), 0)

    def test_multiple_changes(self):
        old_code = "import super_gradients.module1\nfrom super_gradients.module2 import function1\ndef my_function(param1, param2): pass"
        new_code = "import super_gradients.module3\ndef my_function(param1, param2, param3): pass"

        # Imports not checked in regular modules
        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 0)
        self.assertEqual(len(breaking_changes.required_params_added), 1)

        # Imports checked in  __init__.py
        breaking_changes = extract_code_breaking_changes("__init__.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.imports_removed), 2)
        self.assertEqual(len(breaking_changes.required_params_added), 1)

    def test_single_function(self):
        code = "def add(a, b=5):\n    return a + b"

        functions_signatures = parse_functions_signatures(code)
        self.assertIn("add", functions_signatures)
        function_signature = functions_signatures["add"]

        self.assertIsInstance(function_signature, FunctionSignature)
        self.assertEqual(function_signature.name, "add")
        self.assertEqual(function_signature.line_num, 1)

        self.assertIsInstance(function_signature.params, FunctionParameters)
        self.assertEqual(len(function_signature.params.all), 2)
        self.assertEqual(len(function_signature.params.required), 1)
        self.assertEqual(len(function_signature.params.optional), 1)

    def test_multiple_functions(self):
        code = "def add(a, b): pass\ndef subtract(a, b): pass"
        expected = {
            "add": FunctionSignature(
                name="add",
                line_num=1,
                params=FunctionParameters([FunctionParameter(name="a", has_default=False), FunctionParameter(name="b", has_default=False)]),
            ),
            "subtract": FunctionSignature(
                name="subtract",
                line_num=2,
                params=FunctionParameters([FunctionParameter(name="a", has_default=False), FunctionParameter(name="b", has_default=False)]),
            ),
        }
        self.assertEqual(parse_functions_signatures(code), expected)

    def test_parse_nested_functions(self):
        """Make sure that we DON'T detect change in nested functions (this is internal implementation, not API change)."""
        code = "def outer():\n    def inner(a): pass"
        expected = {
            "outer": FunctionSignature(name="outer", line_num=1, params=FunctionParameters([])),
        }
        self.assertEqual(parse_functions_signatures(code), expected)

    def test_no_functions(self):
        code = "a = 5"
        expected = {}
        self.assertEqual(parse_functions_signatures(code), expected)

    def test_class_removed(self):
        old_code = "class MyClass:\n    def method(self): pass"
        new_code = ""

        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.classes_removed), 1)
        self.assertEqual(breaking_changes.classes_removed[0].class_name, "MyClass")

        self.assertEqual(len(breaking_changes.functions_removed), 1)
        self.assertEqual(breaking_changes.functions_removed[0].function_name, "MyClass.method")

    def test_class_methods(self):
        old_code = "class MyClass:\n    def method(self): pass"
        new_code = "class MyClass:\n    def new_method(self): pass"

        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.functions_removed), 1)
        self.assertEqual(breaking_changes.functions_removed[0].function_name, "MyClass.method")

    def test_inheritance_detection(self):
        code = """
class Base1:
    def method_in_base1(self): pass

class Base2:
    def method_in_base2(self): pass

class Derived(Base1, Base2):
    def method_in_derived(self): pass

class DerivedChild(Derived):
    def method_in_derived_child(self): pass
"""
        expected = {
            "Base1.method_in_base1": FunctionSignature(
                name="Base1.method_in_base1", line_num=3, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
            "Base2.method_in_base2": FunctionSignature(
                name="Base2.method_in_base2", line_num=6, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
            "Derived.method_in_base1": FunctionSignature(
                name="Derived.method_in_base1", line_num=3, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
            "Derived.method_in_base2": FunctionSignature(
                name="Derived.method_in_base2", line_num=6, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
            "Derived.method_in_derived": FunctionSignature(
                name="Derived.method_in_derived", line_num=9, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
            "DerivedChild.method_in_base1": FunctionSignature(
                name="DerivedChild.method_in_base1", line_num=3, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
            "DerivedChild.method_in_base2": FunctionSignature(
                name="DerivedChild.method_in_base2", line_num=6, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
            "DerivedChild.method_in_derived": FunctionSignature(
                name="DerivedChild.method_in_derived", line_num=9, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
            "DerivedChild.method_in_derived_child": FunctionSignature(
                name="DerivedChild.method_in_derived_child", line_num=12, params=FunctionParameters([FunctionParameter(name="self", has_default=False)])
            ),
        }
        self.assertEqual(parse_functions_signatures(code), expected)

    def test_inheritance_breaking_change(self):
        old_code = "class MyClass:\n    def method(self): pass"
        new_code = "class MyNewClass:\n    def method(self): pass\nclass MyClass(MyNewClass): pass"  # MyClass.method() will still work

        breaking_changes = extract_code_breaking_changes("module.py", old_code, new_code)
        self.assertEqual(len(breaking_changes.functions_removed), 0)


if __name__ == "__main__":
    unittest.main()
