import itertools

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.datasets.samplers.class_balanced_sampler import ClassBalancedSampler
from super_gradients.training.transforms import DetectionPaddedRescale
from super_gradients.training.utils.collate_fn import DetectionCollateFN


def _get_class_frequency(dataloader):
    class_frequency = {}
    for batch_x, batch_y in tqdm(dataloader, total=len(dataloader)):
        for cls in batch_y[:, -1]:
            cls = int(cls.item())
            class_frequency[cls] = class_frequency.get(cls, 0) + 1

    return class_frequency


def main():
    dataset = COCOFormatDetectionDataset(
        data_dir="/data/coco/",
        images_dir="images/val2017",
        json_annotation_file="annotations/instances_val2017.json",
        transforms=[DetectionPaddedRescale(64)],
        with_crowd=False,
    )

    initial_balance = dataset.get_dataset_classes_information()
    initial_class_balance = initial_balance.sum(axis=0)

    initial_discrepancy = initial_class_balance.max() / initial_class_balance.min()

    print(
        f"BEFORE BALANCE:"
        f" Most frequent class (#{np.argmax(initial_class_balance)}) appears {initial_class_balance.max()} times, which is {initial_discrepancy:.2f}x"
        f" more frequent than the least frequent class (#{np.argmin(initial_class_balance)}) that appears only {initial_class_balance.min()} times!"
    )

    vanilla_dataloader = DataLoader(dataset, batch_size=8, drop_last=False, collate_fn=DetectionCollateFN())
    vanilla_sampled_class_balance = np.zeros_like(initial_class_balance)

    for k, v in _get_class_frequency(vanilla_dataloader).items():
        vanilla_sampled_class_balance[k] = v

    np.testing.assert_equal(vanilla_sampled_class_balance, initial_class_balance)  # no special sampling

    for setting in itertools.product([None, 1.0], [0.5, 1.0, 1.5]):
        oversample_threshold, oversample_agressiveness = setting
        sampler = ClassBalancedSampler(dataset, oversample_threshold=oversample_threshold, oversample_aggressiveness=oversample_agressiveness)
        balanced_dataloader = DataLoader(dataset, batch_size=8, drop_last=False, collate_fn=DetectionCollateFN(), sampler=sampler)
        balanced_sampled_class_balance = np.zeros_like(initial_class_balance)

        for k, v in _get_class_frequency(balanced_dataloader).items():
            balanced_sampled_class_balance[k] = v

        balanced_discrepancy = balanced_sampled_class_balance.max() / balanced_sampled_class_balance.min()

        print(
            f"AFTER BALANCE ({oversample_threshold=}, {oversample_agressiveness=}):"
            f" Most frequent class (#{np.argmax(balanced_sampled_class_balance)}) appears {balanced_sampled_class_balance.max()} times,"
            f" which is {balanced_discrepancy:.2f}x more frequent than the least frequent class (#{np.argmin(balanced_sampled_class_balance)})"
            f" that appears only {balanced_sampled_class_balance.min()} times!"
        )


if __name__ == "__main__":
    main()
