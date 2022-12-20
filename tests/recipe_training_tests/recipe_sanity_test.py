from coverage.annotate import os
from omegaconf import DictConfig
import hydra
import pkg_resources
from super_gradients.common.environment import environment_config
import torch
from super_gradients import Trainer, init_trainer
from super_gradients.common.environment.ddp_utils import multi_process_safe


@multi_process_safe
def _assert_recipe_metric(experiment_name: str, metric_value: float):
    ckpt_dir = os.path.join(environment_config.PKG_CHECKPOINTS_DIR, experiment_name)
    sd = torch.load(os.path.join(ckpt_dir, "ckpt_best.pth"))
    assert sd["acc"].cpu().item() >= metric_value


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name="cifar10_resnet", version_base="1.2")
def main(cfg: DictConfig) -> None:
    goal_metric_val = cfg["goal_metric_val"]
    experiment_name = cfg["experiment_name"]
    Trainer.train_from_config(cfg)
    _assert_recipe_metric(experiment_name, goal_metric_val)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
