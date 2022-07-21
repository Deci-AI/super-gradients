import os

import pandas
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pandas_profiling import ProfileReport


class EDA:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader = None):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader

        self._batch_size = self._val_loader.batch_size

    def _dataloader_to_dataframe(self):
        it = iter(self._val_loader)
        all_classes = set()

        print('Starting DL to DF')
        while True:
            try:
                _, label = next(it)
                for label in torch.unique(label):
                    all_classes.add(label.item())
            except StopIteration:
                break
        print(f"Got {len(all_classes)} unique classes")
        # columns = [f"class_{str(i)}" for i in range(len(all_classes))]
        if os.path.isfile('temp5_csv'):
            df = pandas.read_csv('temp_csv')
        else:
            columns = [f'Class {i}' for i in all_classes]
            columns = ['image'] + columns
            df = pandas.DataFrame(columns=columns)
            for i, (images, labels) in enumerate(self._val_loader):
                if i > 35:
                    break
                for image, label in zip(images, labels):
                    row = [0] * len(all_classes)
                    row[int(label.item())] = 1
                    row = [image] + row
                    df.loc[len(df.index)] = row

            df.to_csv('temp_csv', columns=columns)

        print(df.describe())

        profile = ProfileReport(df, title="Pandas Profiling Report")
        profile.to_file("your_report.html")

    def run(self):
        self._dataloader_to_dataframe()


def get_classification_loaders():
    # training_data = datasets.FashionMNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=ToTensor()
    # )

    val_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # train_dataloader = DataLoader(training_data, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=True)
    train_dataloader = None
    return train_dataloader, val_dataloader


def get_detection_loaders():
    # training_data = datasets.VOCDetection(
    #     image_set='train',
    #     root="data",
    #     download=True,
    #     transform=ToTensor()
    # )

    val_data = datasets.VOCDetection(
        image_set="val",
        root="data",
        download=True,
        transform=ToTensor()
    )

    # train_dataloader = DataLoader(training_data, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=True)
    train_dataloader = None
    return train_dataloader, val_dataloader


def main():
    train_loader, val_loader = get_classification_loaders()
    # train_loader, val_loader = get_detection_loaders()
    eda = EDA(train_loader, val_loader)
    eda.run()


if __name__ == '__main__':
    main()