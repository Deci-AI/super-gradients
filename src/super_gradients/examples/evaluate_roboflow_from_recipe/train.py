# NOT WORKING
import subprocess

import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Name of the hydra config file to use for every run", required=True)
    parser.add_argument("--dataset-names", "-n", nargs="+", help="List of dataset names to run")
    parser.add_argument("--max_epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    config_name = args.config_name
    dataset_names = args.dataset_names
    if not dataset_names:
        pass  # TOODO

    for i, dataset_name in enumerate(dataset_names):
        print(f"\n\n\n> [{i}/100] {dataset_name}\n\n")
        command = (
            f"{sys.executable} "
            f"/home/louis.dupont/PycharmProjects/super-gradients/src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py "
            f"--config-name={config_name} dataset_name={dataset_name}"
        )
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
