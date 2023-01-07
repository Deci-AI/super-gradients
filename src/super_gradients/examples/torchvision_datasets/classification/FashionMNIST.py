import pkg_resources
import hydra

from omegaconf import DictConfig
import torchvision

from super_gradients import Trainer, init_trainer
from super_gradients.training import dataloaders


def run():
    init_trainer()
    main()


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name="user_recipe_mnist_example", version_base="1.2")
def main(cfg: DictConfig) -> None:
    from torchvision import transforms

    train_dataloader = dataloaders.get(
        dataset=torchvision.datasets.FashionMNIST(root="/home/louis.dupont/data", train=True, download=True, transform=transforms.ToTensor()),
        dataloader_params={"batch_size": 10},
    )
    val_dataloader = dataloaders.get(
        dataset=torchvision.datasets.FashionMNIST(root="/home/louis.dupont/data", train=False, download=True, transform=transforms.ToTensor()),
        dataloader_params={"batch_size": 10},
    )

    cfg.train_dataloader = train_dataloader
    cfg.val_dataloader = val_dataloader
    Trainer.train_from_config(cfg)


if __name__ == "__main__":
    run()
