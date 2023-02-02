import unittest

from super_gradients.training.utils.utils import recursive_override


class TestTrainingUtils(unittest.TestCase):
    def test_recursive_override(self):
        base_dict = {"a": 1, "b": 2, "c": {"x": 10, "y": 20, "z": {"q": "q_str", "i": "i_str"}}}

        ext_dict = {"b": 4, "c": {"x": 20, "z": {"q": "q_str_new"}}}

        recursive_override(base_dict, ext_dict)

        self.assertEqual(base_dict["a"], 1)
        self.assertEqual(base_dict["b"], 4)
        self.assertEqual(base_dict["c"]["x"], 20)
        self.assertEqual(base_dict["c"]["y"], 20)
        self.assertEqual(base_dict["c"]["z"]["q"], "q_str_new")
        self.assertEqual(base_dict["c"]["z"]["i"], "i_str")


if __name__ == "__main__":
    unittest.main()
