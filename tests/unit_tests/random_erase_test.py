import torch
import unittest
from super_gradients.training.datasets.data_augmentation import RandomErase


class RandomEraseTest(unittest.TestCase):
    def test_random_erase(self):
        dummy_input = torch.randn(1, 3, 32, 32)
        one_erase = RandomErase(probability=0, value="1.")
        self.assertEqual(one_erase.p, 0)
        self.assertEqual(one_erase.value, 1.0)
        one_erase(dummy_input)

        rndm_erase = RandomErase(probability=0, value="random")
        self.assertEqual(rndm_erase.value, "random")
        rndm_erase(dummy_input)


if __name__ == "__main__":
    unittest.main()
