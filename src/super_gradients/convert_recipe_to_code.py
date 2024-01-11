"""
Entry point for converting recipe file to self-contained train.py file.

Convert a recipe YAML file to a self-contained <train.py> file that can be run with python <train.py>.
Generated file will contain all training hyperparameters from input recipe file but will be self-contained (no dependencies on original recipe).

Limitations: Converting a recipe with command-line overrides of some parameters in this recipe is not supported.

General use: python -m super_gradients.convert_recipe_to_code DESIRED_RECIPE OUTPUT_SCRIPT
Example:     python -m super_gradients.convert_recipe_to_code coco2017_yolo_nas_s train_coco2017_yolo_nas_s.py

For recipe's specific instructions and details refer to the recipe's configuration file in the recipes' directory.
"""
import argparse
import collections
import os.path
import pathlib
from typing import Tuple, Mapping, Dict, Union, Optional, Any, List

import hydra
import pkg_resources
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, ListConfig

from super_gradients import Trainer
from super_gradients.common import MultiGPUMode
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.omegaconf_utils import register_hydra_resolvers
from super_gradients.common.environment.path_utils import normalize_path
from super_gradients.training.utils import get_param

logger = get_logger(__name__)


def try_import_black():
    """
    Attempts to import black code formatter.
    If black is not installed, it will attempt to install it with pip.
    If installation fails, it will return None
    """
    try:
        import black

        return black
    except ImportError:
        logger.info("Trying to install black using pip to enable formatting of the generated script.")
        try:
            import pip

            pip.main(["install", "black==22.10.0"])
            import black

            logger.info("Black installed via pip. ")
            return black
        except Exception:
            logger.info("Black installation failed. Formatting of the generated script will be disabled.")
            return None


def recursively_wrap_strings_with_quotes(cfg):
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if key != "_target_":
                cfg[key] = recursively_wrap_strings_with_quotes(value)
    elif isinstance(cfg, ListConfig):
        for index, item in enumerate(cfg):
            cfg[index] = recursively_wrap_strings_with_quotes(item)
    elif isinstance(cfg, str):
        cfg = _wrap_in_quotes_if_string(cfg)
    return cfg


def recursively_walk_and_extract_hydra_targets(
    cfg: DictConfig, objects: Optional[Mapping] = None, prefix: Optional[str] = None
) -> Tuple[DictConfig, Dict[str, Mapping], List[str]]:
    """
    Iterates over the input config, extracts all hydra targets present in it and replace them with variable references.
    Extracted hydra targets are stored in the objects dictionary (Used to generated instantiations of the objects in the generated script).

    :param cfg:     Input config
    :param objects: Dictionary of extracted hydra targets
    :param prefix:  A prefix variable to track the path to the current config (Used to give variables meaningful name)
    :return:        A tuple of three items (config, hydra targets, additional imports):
                    config - A new config and the dictionary of objects that must be created in the generated script
                    hydra targets - List of hydra targets that should be instantiated from the code
                    additional imports - list of additional imports that should be included in the script
    """
    if objects is None:
        objects = collections.OrderedDict()
    if prefix is None:
        prefix = ""

    additional_imports = []

    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            value, objects, imports = recursively_walk_and_extract_hydra_targets(value, objects, prefix=f"{prefix}_{key}")
            cfg[key] = value
            additional_imports.extend(imports)

        if "_target_" in cfg:
            target_class = cfg["_target_"]
            target_params = dict([(k, _wrap_in_quotes_if_string(v)) for k, v in cfg.items() if k != "_target_"])
            object_name = f"{prefix}".replace(".", "_").lower()
            objects[object_name] = (target_class, target_params)
            import_statement = extract_import_statement_from_hydra_target(target_class)
            cfg = _wrap_variable_reference(object_name)
            additional_imports.append(import_statement)

    elif isinstance(cfg, ListConfig):
        for index, item in enumerate(cfg):
            item, objects, imports = recursively_walk_and_extract_hydra_targets(item, objects, prefix=f"{prefix}_{index}")
            additional_imports.extend(imports)
            cfg[index] = item
    else:
        pass

    return _unwrap_variable_references(cfg), objects, list(set(additional_imports))


def _wrap_variable_reference(variable_name: str) -> Tuple[str, str]:
    """
    Wrap a variable name with a tuple to indicate that it is a reference to a variable.
    It is used to discriminate between "normal" strings and references to instantiated objects.
    We need to be able to distinguish between them because for strings we need to wrap them in quotes in the generated script.
    :param variable_name: Name of the variable
    :return:              A tuple that indicates that this is a reference to a variable
    """
    return "__reference__", variable_name


def _unwrap_variable_references(cfg: DictConfig) -> DictConfig:
    """
    Unwrap variable references from the config. This function modify the configuration in-place.
    This method replace variable reference entries: ("__reference__", object_name) -> object_name
    """
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            cfg[key] = _unwrap_variable_references(value)
    elif isinstance(cfg, (ListConfig, tuple)) and len(cfg) == 2 and cfg[0] == "__reference__":
        return cfg[1]
    elif isinstance(cfg, ListConfig):
        for index, item in enumerate(cfg):
            cfg[index] = _unwrap_variable_references(item)

    return cfg


def extract_import_statement_from_hydra_target(path: str) -> str:
    """
    Locate an object by name or dotted path and return the import statement for it.
    If a hydra target is "numpy.arange" then in order to instantiate it via code we should
    import "numpy" (top-level path) and this method extracts root path and returns it.

    Most of the logic borrowed from hydra._internal.utils._locate
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module

    parts = path.split(".")
    for part in parts:
        if not len(part):
            raise ValueError(
                f"A recipe you are trying to export contains relative imports in hydra target instantiation.\n"
                f"Relative imports are not supported. Problematic import: {path}.\n"
                f"Please change that to absolute import."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        import_module(part0)
    except Exception as exc_import:
        raise ImportError(f"Error loading '{path}':\n{repr(exc_import)}. Are you sure that module '{part0}' is installed?") from exc_import

    return part0


def _wrap_in_quotes_if_string(input: Any) -> Any:
    """
    Append quotes if the input is a string.
    """
    if input is not None and isinstance(input, str):
        return f'"{input}"'
    return input


def convert_recipe_to_code(config_name: Union[str, pathlib.Path], config_dir: Union[str, pathlib.Path], output_script_path: Union[str, pathlib.Path]) -> None:
    """
    Convert a recipe YAML file to a self-contained <train.py> file that can be run with python <train.py>.
    Generated file will contain all training hyperparameters from input recipe file but will be self-contained (no dependencies on original recipe).

    Limitations: Converting a recipe with command-line overrides of some paramters in this recipe is not supported.

    :param config_name:        Name of the recipe file (can be with or without .yaml extension)
    :param config_dir:         Directory where the recipe file is located
    :param output_script_path: Path to the output .py file
    :return:                   None
    """
    config_name = str(config_name)
    config_dir = str(config_dir)
    output_script_path = str(output_script_path)

    register_hydra_resolvers()
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=normalize_path(config_dir), version_base="1.2"):
        cfg = hydra.compose(config_name=config_name)

    cfg = Trainer._trigger_cfg_modifying_callbacks(cfg)
    OmegaConf.resolve(cfg)

    device = get_param(cfg, "device")
    multi_gpu = get_param(cfg, "multi_gpu")

    if multi_gpu is False:
        multi_gpu = MultiGPUMode.OFF
    num_gpus = get_param(cfg, "num_gpus")

    train_dataloader = get_param(cfg, "train_dataloader")
    train_dataset_params = OmegaConf.to_container(cfg.dataset_params.train_dataset_params, resolve=True)
    train_dataloader_params = OmegaConf.to_container(cfg.dataset_params.train_dataloader_params, resolve=True)

    val_dataloader = get_param(cfg, "val_dataloader")
    val_dataset_params = OmegaConf.to_container(cfg.dataset_params.val_dataset_params, resolve=True)
    val_dataloader_params = OmegaConf.to_container(cfg.dataset_params.val_dataloader_params, resolve=True)

    num_classes = cfg.arch_params.num_classes
    arch_params = OmegaConf.to_container(cfg.arch_params, resolve=True)

    cfg.checkpoint_params = hydra.utils.instantiate(cfg.checkpoint_params)

    strict_load = cfg.checkpoint_params.strict_load
    if isinstance(strict_load, Mapping) and "_target_" in strict_load:
        strict_load = hydra.utils.instantiate(strict_load)

    training_hyperparams, hydra_instantiated_objects, additional_imports = recursively_walk_and_extract_hydra_targets(cfg.training_hyperparams)

    checkpoint_num_classes = get_param(cfg.checkpoint_params, "checkpoint_num_classes")
    content = """
import super_gradients
from super_gradients import init_trainer, Trainer
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training import models, dataloaders
from super_gradients.common.data_types.enum import MultiGPUMode, StrictLoad
"""

    for import_statement in additional_imports:
        content += f"import {import_statement}\n"

    content += f"""
def main():
    init_trainer()
    setup_device(device={device}, multi_gpu="{multi_gpu}", num_gpus={num_gpus})

    trainer = Trainer(experiment_name="{cfg.experiment_name}", ckpt_root_dir="{cfg.ckpt_root_dir}")

    num_classes = {num_classes}
    arch_params = {arch_params}

    model = models.get(
        model_name="{cfg.architecture}",
        num_classes=num_classes,
        arch_params=arch_params,
        strict_load={_wrap_in_quotes_if_string(strict_load)},
        pretrained_weights={_wrap_in_quotes_if_string(cfg.checkpoint_params.pretrained_weights)},
        checkpoint_path={_wrap_in_quotes_if_string(cfg.checkpoint_params.checkpoint_path)},
        load_backbone={cfg.checkpoint_params.load_backbone},
        checkpoint_num_classes={checkpoint_num_classes},
    )

    train_dataloader = dataloaders.get(
        name={_wrap_in_quotes_if_string(train_dataloader)},
        dataset_params={train_dataset_params},
        dataloader_params={train_dataloader_params},
    )

    val_dataloader = dataloaders.get(
        name={_wrap_in_quotes_if_string(val_dataloader)},
        dataset_params={val_dataset_params},
        dataloader_params={val_dataloader_params},
    )

"""
    for name, (class_name, class_params) in hydra_instantiated_objects.items():
        class_params_str = []
        for k, v in class_params.items():
            class_params_str.append(f"{k}={v}")
        class_params_str = ",".join(class_params_str)
        content += f"    {name} = {class_name}({class_params_str})\n\n"

    content += f"""

    training_hyperparams = {training_hyperparams}

    # TRAIN
    result = trainer.train(
        model=model,
        train_loader=train_dataloader,
        valid_loader=val_dataloader,
        training_params=training_hyperparams,
    )

    print(result)

if __name__ == "__main__":
    main()
"""
    # Remove quotes from dict values to reference them as variables
    for key in hydra_instantiated_objects.keys():
        key_to_search = f"'{key}'"
        key_to_replace_with = f"{key}"
        content = content.replace(key_to_search, key_to_replace_with)

    with open(output_script_path, "w") as f:
        black = try_import_black()
        if black is not None:
            content = black.format_str(content, mode=black.FileMode(line_length=160))
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help=".yaml filename")
    parser.add_argument("save_path", type=str, default=None, help="Destination path to the output .py file")
    parser.add_argument("--config_dir", type=str, default=pkg_resources.resource_filename("super_gradients.recipes", ""), help="The config directory path")
    args = parser.parse_args()

    save_path = args.save_path or os.path.splitext(os.path.basename(args.config_name))[0] + ".py"
    logger.info(f"Saving recipe script to {save_path}")

    convert_recipe_to_code(args.config_name, args.config_dir, save_path)


if __name__ == "__main__":
    main()
