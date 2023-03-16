import unittest
from super_gradients.training import models


class MyTestCase(unittest.TestCase):
    def test_something(self):
        model = models.get("deciyolo_s", num_classes=8)
        print(model)


if __name__ == "__main__":
    unittest.main()
