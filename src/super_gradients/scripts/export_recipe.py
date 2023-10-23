"""
Script that saves a exportd (i.e no inheritance from other yaml configuration files),
 .yaml file that can be ran on its own without the need to keep other configurations which the original
  file inherits from.

Usage:
    python export_recipe --config_name=cifar10_resnet -> saves cifar10_resnet_complete.yaml in path/to/super_gradients/recipes

    python export_recipe --config_dir=/path/to/recipes/ config_name=my_recipe.yaml -> saves config_name_complete.yaml in /path/to/recipes/

    python export_recipe --config_dir=/path/to/recipes/ config_name=my_recipe.yaml save_path=/other/recipes/dir/my_complete_recipe.yaml
     -> saves the complete receip in /other/recipes/dir/my_complete_recipe.yaml

:arg config_name: The .yaml config filename (can leave the .yaml postfix out, but not mandatory).

:arg config_dir: The config directory path, as absolute file system path.
 When None, will use SG's recipe directory (i.e path/to/super_gradients/recipes)

:arg: The output path for the complete .yaml file.
 When None, will use config_dir/config_name with a trailing "_complete.yaml".
"""

import argparse
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
import pkg_resources
from omegaconf import OmegaConf
import os
from super_gradients import init_trainer
from super_gradients.common.environment.cfg_utils import normalize_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default=pkg_resources.resource_filename("super_gradients.recipes", ""), help="The config directory path")
    parser.add_argument("--config_name", type=str, help=".yaml filename")
    parser.add_argument("--save_path", type=str, default=None, help="Destination path to the output .yaml file")
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = os.path.join(args.config_dir, args.config_name).replace(".yaml", "") + "_complete.yaml"

    # NEED TO REGISTER RESOLVERS FIRST
    init_trainer()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=normalize_path(args.config_dir), version_base="1.2"):
        cfg = compose(config_name=args.config_name)
        OmegaConf.save(config=cfg, f=args.save_path)
        print(f"Successfully saved recipe at {args.save_path}")
