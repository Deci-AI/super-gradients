import unittest

from super_gradients.training.utils.config_utils import ConfigInspector, raise_if_unused_params


class ConfigInspectTest(unittest.TestCase):
    def test_inspector_raise_on_unused_args(self):
        def model_factory(cfg):
            return 42

        config = {"unused_param": True}

        with self.assertRaisesRegex("", ""):
            with raise_if_unused_params(config) as config:
                _ = model_factory(config)

    def test_inspector_with_dict_and_list(self):
        config = {
            "beta": 0.73,
            "lr": 1e-4,
            "encoder": {
                "indexes": [1, 2, 3],
                "pretrained": True,
                "backbone": "yolov3",
                "layers": [
                    {"blocks": 4},
                    {"blocks": 3},
                    {"blocks": 6},
                    {"blocks": 9},
                ],
            },
        }

        c = ConfigInspector.wrap(config)

        # Simulate parameters usage
        print(c["beta"])
        print(c["encoder"]["layers"])
        print(sum(c["encoder"]["indexes"]))
        print(c["beta"])
        print(c["encoder"]["layers"][0])
        print(c["encoder"]["layers"][3]["blocks"])

        print("All parameters")
        print(c.all_params)

        print("Unused parameters")
        print(c.unused_params)

        self.assertSetEqual(
            c.unused_params,
            {
                "lr",
                "encoder.pretrained",
                "encoder.backbone",
                "encoder.layers.0.blocks",
                "encoder.layers.1",
                "encoder.layers.1.blocks",
                "encoder.layers.2",
                "encoder.layers.2.blocks",
                "encoder.layers.2.blocks",
            },
        )
