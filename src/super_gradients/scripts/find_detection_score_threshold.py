"""
Find the best confidence score threshold for each class in object detection tasks
Use this script when you have a trained model and want to analyze / optimize its performance
The thresholds can be used later when performing NMS
Usage is similar to src/super_gradients/evaluate_from_recipe.py

Notes:
    This script does NOT run TRAINING, so make sure in the recipe that you load a PRETRAINED MODEL
    either from one of your checkpoint or from a pretrained model.

General use: python -m super_gradients.scripts.find_detection_score_threshold --config-name="DESIRED_RECIPE" architecture="DESIRED_ARCH"
            checkpoint_params.pretrained_weights="DESIRED_DATASET"

Example: python -m super_gradients.scripts.find_detection_score_threshold --config-name=coco2017_yolox architecture=yolox_n
            checkpoint_params.pretrained_weights=coco
"""

import hydra
import pkg_resources
from omegaconf import DictConfig

from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.common.environment.cfg_utils import add_params_to_cfg
from super_gradients import Trainer, init_trainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    add_params_to_cfg(cfg.training_hyperparams.valid_metrics_list[0].DetectionMetrics, params=["calc_best_score_thresholds=True"])
    _, valid_metrics_dict = Trainer.evaluate_from_recipe(cfg)

    class_names = COCO_DETECTION_CLASSES_LIST  # change this line to use a different dataset
    prefix = "Best_score_threshold_cls_"
    best_thresholds = {int(k[len(prefix) :]): v for k, v in valid_metrics_dict.items() if k.startswith(prefix)}
    assert len(best_thresholds) == len(class_names)
    print("-----Best_score_thresholds-----")
    max_class_name = max(len(class_name) for class_name in class_names)
    for k, v in best_thresholds.items():
        print(f"{class_names[k]:<{max_class_name}} (class {k}):\t{v:.2f}")


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
