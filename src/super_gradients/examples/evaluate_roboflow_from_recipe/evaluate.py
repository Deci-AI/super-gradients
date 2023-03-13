from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import Trainer, init_trainer

from super_gradients.training.utils import get_param
from super_gradients.training.utils.distributed_training_utils import (
    setup_device,
)


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    setup_device(
        device=get_param(cfg, "device"),
        multi_gpu=get_param(cfg, "multi_gpu"),
        num_gpus=get_param(cfg, "num_gpus"),
    )

    cfg.multi_gpu = "Off"
    cfg.num_gpus = 1
    from super_gradients.training.datasets.detection_datasets.roboflow100 import ROBOFLOW_DATASETS_NAMES_WITH_CATEGORY

    original_experiment_name = cfg.experiment_name
    for dataset in list(ROBOFLOW_DATASETS_NAMES_WITH_CATEGORY.keys())[12:]:
        cfg.dataset_params.dataset_name = dataset
        cfg.experiment_name = original_experiment_name + "_" + dataset
        print(f"launching: {cfg.experiment_name}")

        Trainer.train_from_config(cfg)
        Trainer.evaluate_checkpoint(experiment_name=cfg.experiment_name, ckpt_name="ckpt_best.pth")
        x, y = Trainer.evaluate_from_recipe(cfg)

        print("--------------------------")
        print(y)
        print("--------------------------")


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
