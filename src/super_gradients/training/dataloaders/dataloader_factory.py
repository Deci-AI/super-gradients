from hydra import initialize, compose
import pkg_resources
# 1. initialize will add config_path the config search path within the context
# 2. The module with your configs should be importable.
#    it needs to have a __init__.py (can be empty).
# 3. THe config path is relative to the file calling initialize (this file)

def test_with_initialize() -> None:
    with initialize(config_path="../../recipes"):
        # config is relative to a module
        cfg = compose(config_name="cifar10_resnet")
        print(cfg)

test_with_initialize()