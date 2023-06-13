import hydra
from omegaconf import omegaconf

from super_gradients import init_trainer
from super_gradients.training import Trainer


# cityscapes_segformer_sophia
# coco2017_pose_dekr_w32_no_dc_sophia
# coco2017_ppyoloe_m_sophia.yaml
@hydra.main(config_path="./super_gradients/recipes", config_name='cityscapes_segformer_sophia', version_base="1.2.0")
def main(cfg: omegaconf.DictConfig) -> None:
    Trainer.train_from_config(cfg)


if __name__ == '__main__':
    init_trainer()
    main()
