import argparse
import hydra
from super_gradients import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("config_name", type=str)
args = parser.parse_args()

config_name = args.config_name

with hydra.initialize(config_path="recipes"):
    cfg = hydra.compose(config_name=config_name)
Trainer.train_from_config(cfg)
