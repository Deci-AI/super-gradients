import torch
import unittest

from super_gradients.training.models import ShelfNet18_LW, ShelfNet34_LW


class TestShelfNet(unittest.TestCase):
    def test_shelfnet_creation(self):
        """
        test_shelfnet_creation - Tests the creation of the model class itself
            :return:
        """
        dummy_input = torch.randn(1, 3, 512, 512)
        shelfnet18_model = ShelfNet18_LW(num_classes=21)
        shelfnet34_model = ShelfNet34_LW(num_classes=21)

        # FIXME: FIX MODEL FORWARD TESTING FOR SHELFNET50 and 101
        for model in [shelfnet18_model, shelfnet34_model]:
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
                self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
