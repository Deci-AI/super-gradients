import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset

from super_gradients import setup_device
from super_gradients.training import dataloaders


def get_dataset(dataset_size, image_size):
    images = torch.Tensor(np.zeros((dataset_size, 3, image_size, image_size)))
    ground_truth = torch.LongTensor(np.zeros((dataset_size)))
    dataset = TensorDataset(images, ground_truth)
    return dataset


if __name__ == "__main__":

    setup_device(
        device="cuda",
        multi_gpu="DDP",
        num_gpus=4,
    )

    dataset = get_dataset(dataset_size=64, image_size=32)
    dataloader = dataloaders.get(dataset=dataset, dataloader_params={"batch_size": 4, "min_samples": 80, "drop_last": True})

    if len(dataloader) == 5:
        torch.distributed.destroy_process_group()
        sys.exit(0)
    else:
        print(f"wrong datalaoder length, expected min_samples/(world_size*batch_size)=80/(4*4=5), got {len(dataloader)}")
        torch.distributed.destroy_process_group()
        sys.exit(1)
