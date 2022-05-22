"""
Cifar10 training with SuperGradients training with the following initialized torch objects:

    DataLoaders
    Optimizers
    Networks (nn.Module)
    Schedulers
    Loss functions

Main purpose is to demonstrate training in SG with minimal abstraction and maximal flexibility
"""
from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import SgModel
import super_gradients
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import get_data_loader, get_eval_loader
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training import MultiGPUMode
from super_gradients.training.utils.callbacks import YoloXTrainingStageSwitchCallback

from super_gradients.training.datasets.datasets_utils import MultiscaleForwardPassPrepFunction

from loguru import logger


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""))
@logger.catch
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)

    train_loader = get_data_loader(cfg)
    valid_loader = get_eval_loader(cfg)
    classes = COCO_DETECTION_CLASSES_LIST
    # train_loader, valid_loader, classes = None, None, None
    cfg.sg_model = SgModel(cfg.sg_model.experiment_name, cfg.model_checkpoints_location,
                           train_loader=train_loader, valid_loader=valid_loader, classes=classes,
                           multi_gpu=MultiGPUMode(cfg.multi_gpu))

    # cfg.sg_model.connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)
    cfg.sg_model.build_model(cfg.architecture, arch_params=cfg.arch_params, checkpoint_params=cfg.checkpoint_params)

    # cfg.training_hyperparams.initial_lr /= 64
    # cfg.training_hyperparams.initial_lr *= cfg.dataset_params.batch_size * 8
    # dvcb = DetectionVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
    #                                       freq=1,
    #                                       post_prediction_callback=YoloV5PostPredictionCallback(iou=0.65, conf=0.99),
    #                                       classes=classes,
    #                                       last_img_idx_in_batch=8)
    cfg.training_hyperparams.forward_pass_prep_fn = MultiscaleForwardPassPrepFunction()
    cfg.training_hyperparams.phase_callbacks = [YoloXTrainingStageSwitchCallback(285)]#, YoloXMultiscaleImagesCallback()]
    print(cfg.training_hyperparams.initial_lr)

    cfg.sg_model.train(training_params=cfg.training_hyperparams)


if __name__ == "__main__":
    super_gradients.init_trainer()
    main()


