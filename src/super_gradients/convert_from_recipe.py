"""
Entry point for converting recipe file to self-contained train.py file.

General use: python -m super_gradients.convert_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
import argparse
import collections
import os.path
from typing import Tuple, Mapping, Dict

import black
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


def recursively_walk_and_extract_hydra_targets(cfg, objects=None, prefix=None) -> Tuple[DictConfig, Dict[str, Mapping]]:
    if objects is None:
        objects = collections.OrderedDict()
    if prefix is None:
        prefix = ""

    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            value, objects = recursively_walk_and_extract_hydra_targets(value, objects, prefix=f"{prefix}_{key}")
            cfg[key] = value

        if "_target_" in cfg:
            target_class = cfg["_target_"]
            target_params = dict([(k, v) for k, v in cfg.items() if k != "_target_"])
            object_name = f"{prefix}".replace(".", "_").lower()
            objects[object_name] = (target_class, target_params)
            cfg = object_name

    elif isinstance(cfg, ListConfig):
        for index, item in enumerate(cfg):
            item, objects = recursively_walk_and_extract_hydra_targets(item, objects, prefix=f"{prefix}_{index}")
            cfg[index] = item
    else:
        print(f"Skipping {cfg}")
        pass
    return cfg, objects


def convert_from_recipe(cfg: DictConfig, output_script_path: str):
    cfg = Trainer._trigger_cfg_modifying_callbacks(cfg)

    device = get_param(cfg, "device")
    multi_gpu = get_param(cfg, "multi_gpu")

    if multi_gpu is False:
        multi_gpu = MultiGPUMode.OFF
    num_gpus = get_param(cfg, "num_gpus")
    print(device, multi_gpu, num_gpus)

    train_dataloader = get_param(cfg, "train_dataloader")
    train_dataset_params = OmegaConf.to_container(cfg.dataset_params.train_dataset_params, resolve=True)
    train_dataloader_params = OmegaConf.to_container(cfg.dataset_params.train_dataloader_params, resolve=True)

    val_dataloader = get_param(cfg, "val_dataloader")
    val_dataset_params = OmegaConf.to_container(cfg.dataset_params.val_dataset_params, resolve=True)
    val_dataloader_params = OmegaConf.to_container(cfg.dataset_params.val_dataloader_params, resolve=True)

    num_classes = cfg.arch_params.num_classes
    arch_params = OmegaConf.to_container(cfg.arch_params, resolve=True)

    training_hyperparams, hydra_instantiated_objects = recursively_walk_and_extract_hydra_targets(cfg.training_hyperparams)
    print(hydra_instantiated_objects)

    checkpoint_num_classes = get_param(cfg.checkpoint_params, "checkpoint_num_classes")
    content = f"""
import super_gradients
from super_gradients import init_trainer, Trainer
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training import models, dataloaders
from super_gradients.common.data_types.enum import MultiGPUMode, StrictLoad
import numpy as np

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
        strict_load={hydra.utils.instantiate(cfg.checkpoint_params.strict_load)},
        pretrained_weights={cfg.checkpoint_params.pretrained_weights},
        checkpoint_path={cfg.checkpoint_params.checkpoint_path},
        load_backbone={cfg.checkpoint_params.load_backbone},
        checkpoint_num_classes={checkpoint_num_classes},
    )

    train_dataloader = dataloaders.get(
        name="{train_dataloader}",
        dataset_params={train_dataset_params},
        dataloader_params={train_dataloader_params},
    )

    val_dataloader = dataloaders.get(
        name="{val_dataloader}",
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
        print(key_to_search, key_to_replace_with, key_to_search in content)
        content = content.replace(key_to_search, key_to_replace_with)

    with open(output_script_path, "w") as f:
        content = black.format_str(content, mode=black.FileMode(line_length=120))
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help=".yaml filename")
    parser.add_argument("--config_dir", type=str, default=pkg_resources.resource_filename("super_gradients.recipes", ""), help="The config directory path")
    parser.add_argument("--save_path", type=str, default=None, help="Destination path to the output .py file")
    args = parser.parse_args()

    save_path = args.save_path or os.path.splitext(os.path.basename(args.config_name))[0] + ".py"
    logger.info(f"Saving recipe script to {save_path}")

    register_hydra_resolvers()
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=normalize_path(args.config_dir), version_base="1.2"):
        cfg = hydra.compose(config_name=args.config_name)

    convert_from_recipe(cfg, save_path)


if __name__ == "__main__":
    main()
