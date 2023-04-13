from omegaconf import DictConfig
import hydra
import pkg_resources

from super_gradients import Trainer, init_trainer


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name="script_generate_rescoring_data", version_base="1.2")
def main(cfg: DictConfig) -> None:
    Trainer.train_from_config(cfg)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
