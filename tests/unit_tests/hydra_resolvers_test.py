import unittest

from omegaconf import OmegaConf

from super_gradients.common.environment.ddp_utils import register_hydra_resolvers


class HydraResolversTest(unittest.TestCase):
    def setUp(self) -> None:
        register_hydra_resolvers()

    def test_add(self):
        conf = OmegaConf.create({"a": 1, "b": 2, "c": 3, "a_plus_b": "${add: ${a},${b}}", "a_plus_b_plus_c": "${add: ${a}, ${b}, ${c}}"})
        self.assertEqual(conf["a_plus_b"], 3)
        self.assertEqual(conf["a_plus_b_plus_c"], 6)

    def test_div(self):
        conf = OmegaConf.create({"a": 1, "b": 2, "a_div_b": "${div: ${a},${b}}"})
        self.assertAlmostEqual(conf["a_div_b"], 0.5)

    def test_mul(self):
        conf = OmegaConf.create({"a": 1, "b": 2, "c": 4, "a_mul_b": "${mul: ${a},${b}}", "a_mul_b_mul_c": "${mul: ${a}, ${b}, ${c}}"})
        self.assertEqual(conf["a_mul_b"], 2)
        self.assertEqual(conf["a_mul_b_mul_c"], 8)

    def test_list(self):
        conf = OmegaConf.create(
            {
                "my_list": [10, 20, 30, 40, 50],
                "third_of_list": "${getitem: ${my_list}, 2}",
                "first_of_list": "${first: ${my_list}}",
                "last_of_list": "${last: ${my_list}}",
            }
        )
        self.assertEqual(conf["third_of_list"], 30)
        self.assertEqual(conf["first_of_list"], 10)
        self.assertEqual(conf["last_of_list"], 50)

    def test_cond(self):
        conf = OmegaConf.create(
            {
                "boolean": True,
                "a": "red_pill",
                "b": "blue_pill",
                "result": "${cond:${boolean}, ${a}, ${b}}",
            }
        )
        assert conf["result"] == "red_pill"

        conf["boolean"] = False
        assert conf["result"] == "blue_pill"
