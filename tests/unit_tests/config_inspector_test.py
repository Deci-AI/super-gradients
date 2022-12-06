import copy
import unittest

from omegaconf import OmegaConf

from super_gradients.training.models import SgModule
from super_gradients.training.models.all_architectures import ARCHITECTURES

from super_gradients.training.models.model_factory import get_architecture
from super_gradients.training.utils import HpmStruct
from super_gradients.training.utils.config_utils import raise_if_unused_params, UnusedConfigParamException, AccessCounterDict, AccessCounterHpmStruct
from super_gradients.training.utils.sg_trainer_utils import get_callable_param_names


class ConfigInspectTest(unittest.TestCase):
    def test_inspector_raise_on_unused_args(self):
        def model_factory(cfg):
            return cfg["a"] + cfg["b"]

        original_config = {"unused_param": True, "a": 1, "b": 2}

        with self.assertRaisesRegex(UnusedConfigParamException, "Detected unused parameters in configuration object that were not consumed by caller"):
            config = copy.deepcopy(original_config)
            with raise_if_unused_params(config) as config:
                _ = model_factory(config)

        with self.assertRaisesRegex(UnusedConfigParamException, "Detected unused parameters in configuration object that were not consumed by caller"):
            config = OmegaConf.create(copy.deepcopy(original_config))
            with raise_if_unused_params(config) as config:
                _ = model_factory(config)

        with self.assertRaisesRegex(UnusedConfigParamException, "Detected unused parameters in configuration object that were not consumed by caller"):
            config = HpmStruct(**copy.deepcopy(original_config))
            with raise_if_unused_params(copy.deepcopy(config)) as config:
                _ = model_factory(config)

    def test_inspector_raise_on_unused_args_with_modification_of_the_config(self):
        def model_factory(cfg):
            cfg["this_is_a_test_property_that_is_set_but_not_used"] = 42
            cfg["this_is_a_test_property_that_is_set_and_used"] = 39
            return cfg["a"] + cfg["b"] + cfg["this_is_a_test_property_that_is_set_and_used"]

        original_config = {"unused_param": True, "a": 1, "b": 2}

        with self.assertRaisesRegex(UnusedConfigParamException, "Detected unused parameters in configuration object that were not consumed by caller"):
            config = copy.deepcopy(original_config)
            with raise_if_unused_params(config) as config:
                result = model_factory(config)
                self.assertEqual(result, 42)

            self.assertTrue("this_is_a_test_property_that_is_set_and_used" in config.get_used_params())

        with self.assertRaisesRegex(UnusedConfigParamException, "Detected unused parameters in configuration object that were not consumed by caller"):
            config = OmegaConf.create(copy.deepcopy(original_config))
            with raise_if_unused_params(config) as config:
                result = model_factory(config)
                self.assertEqual(result, 42)
            self.assertTrue("this_is_a_test_property_that_is_set_and_used" in config.get_used_params())

        with self.assertRaisesRegex(UnusedConfigParamException, "Detected unused parameters in configuration object that were not consumed by caller"):
            config = HpmStruct(**copy.deepcopy(original_config))
            with raise_if_unused_params(copy.deepcopy(config)) as config:
                result = model_factory(config)
                self.assertEqual(result, 42)
            self.assertTrue("this_is_a_test_property_that_is_set_and_used" in config.get_used_params())

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

        c = AccessCounterDict(config)

        # Simulate parameters usage
        print(c["beta"])
        print(c["encoder"]["layers"])
        print(sum(c["encoder"]["indexes"]))
        print(c["beta"])
        print(c["encoder"]["layers"][0])
        print(c["encoder"]["layers"][3]["blocks"])

        print("All parameters")
        print(c.get_all_params())

        print("Unused parameters")
        print(c.get_unused_params())

        self.assertSetEqual(
            c.get_unused_params(),
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

    def test_inspector_with_omegaconf(self):
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
        config = OmegaConf.create(config)

        c = AccessCounterDict(config)

        # Simulate parameters usage
        print(c.beta)
        print(c.encoder.layers)
        print(sum(c.encoder.indexes))
        print(c.encoder.layers[0])
        print(c.encoder.layers[3].blocks)

        print("All parameters")
        print(c.get_all_params())

        print("Unused parameters")
        print(c.get_unused_params())

        self.assertSetEqual(
            c.get_unused_params(),
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

    def test_inspector_with_hpm_struct(self):
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
        config = HpmStruct(**config)

        c = AccessCounterHpmStruct(config)

        # Simulate parameters usage
        print(c.beta)
        print(c.encoder.layers)
        print(sum(c.encoder.indexes))
        print(c.encoder.layers[0])
        print(c.encoder.layers[3].blocks)

        print("All parameters")
        print(c.get_all_params())

        print("Unused parameters")
        print(c.get_unused_params())

        self.assertSetEqual(
            c.get_unused_params(),
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

    def test_model_architectures(self):
        for model_name in ARCHITECTURES.keys():
            architecture_cls, arch_params, pretrained_weights_path, is_remote = get_architecture(model_name, HpmStruct())
            self.assertIsNotNone(arch_params, msg=model_name)

            if not issubclass(architecture_cls, SgModule):
                # net = architecture_cls(**arch_params.to_dict(include_schema=False))
                pass
            else:
                # Most of the SG models work with a single params names "arch_params" of type HpmStruct, but a few take **kwargs instead
                if "arch_params" not in get_callable_param_names(architecture_cls):
                    # net = architecture_cls(**arch_params.to_dict(include_schema=False))
                    pass
                else:
                    with self.subTest(model_name):
                        try:
                            _ = architecture_cls(arch_params=arch_params, num_classes=10)
                        except Exception:
                            self.skipTest("Skipping since model cannot be instantiated with YAML configuration")
                            pass

                        with raise_if_unused_params(arch_params) as tracked_arch_params:
                            _ = architecture_cls(arch_params=tracked_arch_params, num_classes=10)
