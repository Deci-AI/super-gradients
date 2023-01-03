import pkg_resources
import hydra
import torch.utils.data.dataset

from omegaconf import DictConfig
import torchvision
import torchvision.transforms as transform

from super_gradients import Trainer, init_trainer
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.utils.sg_trainer_utils import parse_args

from super_gradients.training.utils.distributed_training_utils import setup_device
from omegaconf import OmegaConf


def run():
    init_trainer()
    main()


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name="user_recipe_mnist_example", version_base="1.2")
def main(cfg: DictConfig) -> None:

    setup_device(multi_gpu=core_utils.get_param(cfg, "multi_gpu", MultiGPUMode.OFF), num_gpus=core_utils.get_param(cfg, "num_gpus"))

    # INSTANTIATE ALL OBJECTS IN CFG
    cfg = hydra.utils.instantiate(cfg)

    kwargs = parse_args(cfg, Trainer.__init__)

    trainer = Trainer(**kwargs)

    # INSTANTIATE DATA LOADERS
    train_dataset = wrap_segmentation_dataset(
        dataset=torchvision.datasets.Cityscapes(root="/data/cityscapes", target_type="semantic", split="train"),
        transforms=cfg.dataset_params.train_dataset_params.transforms,
    )
    val_dataset = wrap_segmentation_dataset(
        dataset=torchvision.datasets.Cityscapes(root="/data/cityscapes", target_type="semantic", split="val"),
        transforms=cfg.dataset_params.val_dataset_params.transforms,
    )
    train_dataloader = dataloaders.get(dataset=train_dataset, dataloader_params={"batch_size": 2})
    val_dataloader = dataloaders.get(dataset=val_dataset, dataloader_params={"batch_size": 2})

    #
    # BUILD NETWORK
    model = models.get(
        model_name=cfg.architecture,
        num_classes=cfg.arch_params.num_classes,
        arch_params=cfg.arch_params,
        strict_load=cfg.checkpoint_params.strict_load,
        pretrained_weights=cfg.checkpoint_params.pretrained_weights,
        checkpoint_path=cfg.checkpoint_params.checkpoint_path,
        load_backbone=cfg.checkpoint_params.load_backbone,
    )
    recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}

    # TRAIN
    trainer.train(
        model=model,
        train_loader=train_dataloader,
        valid_loader=val_dataloader,
        training_params=cfg.training_hyperparams,
        additional_configs_to_log=recipe_logged_cfg,
    )


class TransformDict:
    def __init__(self, transforms: dict):
        self.transforms = transforms

    def __call__(self, sample):
        for col, col_transform in self.transforms.items():
            sample[col] = col_transform(sample[col])
        return sample


class CustomDataset(torch.utils.data.dataset.Dataset):
    """Proxy dataset to wrap __getitem__"""

    def __init__(self, dataset, transforms, columns):
        self.dataset = dataset

        self.transforms = transform.Compose(transforms)
        self.columns = columns

    def __getitem__(self, item):
        items = self.dataset[item]
        if len(items) != len(self.columns):
            raise ValueError
        sample = {col: val for col, val in zip(self.columns, items)}
        sample = self.transforms(sample)
        return tuple(sample[col] for col in self.columns)

    def __len__(self):
        return len(self.dataset)


@resolve_param("transforms", factory=TransformsFactory())
def wrap_segmentation_dataset(dataset, transforms):
    import numpy as np

    def process_target(target):
        target = torch.from_numpy(np.array(target)).long()
        target[target == 255] = 19
        return target

    custom_transform = TransformDict(
        {"image": transform.Compose([transform.ToTensor(), transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), "mask": process_target}
    )
    transforms += [custom_transform]
    return CustomDataset(dataset=dataset, transforms=transforms, columns=("image", "mask"))


if __name__ == "__main__":
    run()
