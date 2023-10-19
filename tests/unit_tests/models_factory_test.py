import unittest
from super_gradients.training import models
from super_gradients.common.registry.registry import ARCHITECTURES


class DynamicModelTests(unittest.TestCase):
    def test_model_instantiation(self):
        for model_name in ARCHITECTURES.keys():
            with self.subTest(model_name=model_name):
                _ = models.get(model_name, num_classes=20)


if __name__ == "__main__":
    unittest.main()
