import unittest

from omegaconf import OmegaConf

from super_gradients.common.environment.env_helpers import register_hydra_resolvers


class HydraResolversTest(unittest.TestCase):

    def setUp(self) -> None:
        register_hydra_resolvers()

    def test_add(self):
        conf = OmegaConf.create({
            "a": 1,
            "b": 2,
            "c": 3,
            "a_plus_b": "${add: ${a},${b}}",
            "a_plus_b_plus_c": "${add: ${a}, ${b}, ${c}}"
        })
        assert conf["a_plus_b"] == 3
        assert conf["a_plus_b_plus_c"] == 6

    def test_cond(self):
        conf = OmegaConf.create({
            "boolean": True,
            "a": "red_pill",
            "b": "blue_pill",
            "result": "${cond:${boolean}, ${a}, ${b}}",
        })
        assert conf["result"] == "red_pill"

        conf["boolean"] = False
        assert conf["result"] == "blue_pill"
