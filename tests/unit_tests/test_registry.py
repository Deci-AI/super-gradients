import unittest
from typing import List

from super_gradients.common.registry.registry import create_register_decorator
from super_gradients.common.factories.base_factory import BaseFactory, UnknownTypeException


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        # We do all the registration in `setUp` to avoid having registration ran on import
        _DUMMY_REGISTRY = {}
        register_class = create_register_decorator(registry=_DUMMY_REGISTRY)

        @register_class("good_object_name")
        class Class1:
            def __init__(self, values: List[float]):
                self.values = values

        @register_class(deprecated_name="deprecated_object_name")
        class Class2:
            def __init__(self, values: List[float]):
                self.values = values

        self.Class1 = Class1  # Save classes, not instances
        self.Class2 = Class2
        self.factory = BaseFactory(type_dict=_DUMMY_REGISTRY)

    def test_instantiate_from_name(self):
        instance = self.factory.get({"good_object_name": {"values": [1.0, 2.0]}})
        self.assertIsInstance(instance, self.Class1)

    def test_instantiate_from_classname_when_name_set(self):
        with self.assertRaises(UnknownTypeException):
            self.factory.get({"Class1": {"values": [1.0, 2.0]}})

    def test_instantiate_from_classname_when_no_name_set(self):
        instance = self.factory.get({"Class2": {"values": [1.0, 2.0]}})
        self.assertIsInstance(instance, self.Class2)

    def test_instantiate_from_deprecated_name(self):
        with self.assertWarns(DeprecationWarning):
            instance = self.factory.get({"deprecated_object_name": {"values": [1.0, 2.0]}})
        self.assertIsInstance(instance, self.Class2)


if __name__ == "__main__":
    unittest.main()
