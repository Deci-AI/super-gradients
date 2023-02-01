import os

import torch
import torch.nn as nn
import torch.nn.init as init


def prefetch_dataset(dataset, num_workers=4, batch_size=32, device=None, half=False):
    if isinstance(dataset, list) and isinstance(dataset[0], torch.Tensor):
        tensors = dataset
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=False)
        tensors = [t for t in dataloader]
        tensors = [torch.cat(t, dim=0) for t in zip(*tensors)]

    if device is not None:
        tensors = [t.to(device=device) for t in tensors]
    if half:
        tensors = [t.half() if t.is_floating_point() else t for t in tensors]

    return torch.utils.data.TensorDataset(*tensors)


class PrefetchDataLoader:
    def __init__(self, dataloader, device, half=False):
        self.loader = dataloader
        self.iter = None
        self.device = device
        self.dtype = torch.float16 if half else torch.float32
        self.stream = torch.cuda.Stream()
        self.next_data = None

    def __len__(self):
        return len(self.loader)

    def async_prefech(self):
        try:
            self.next_data = next(self.iter)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            if isinstance(self.next_data, torch.Tensor):
                self.next_data = self.next_data.to(dtype=self.dtype, device=self.device, non_blocking=True)
            elif isinstance(self.next_data, (list, tuple)):
                self.next_data = [
                    t.to(dtype=self.dtype, device=self.device, non_blocking=True) if t.is_floating_point() else t.to(device=self.device, non_blocking=True)
                    for t in self.next_data
                ]

    def __iter__(self):
        self.iter = iter(self.loader)
        self.async_prefech()
        while self.next_data is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            data = self.next_data
            self.async_prefech()
            yield data


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            # if m.bias:
            #    init.constant(m.bias, -5)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def is_better(new_metric, current_best_metric, metric_to_watch="acc"):
    """
    Determines which of the two metrics is better, the higher if watching acc or lower when watching loss
    :param new_metric:                 the new metric
    :param current_best_metric:        the compared to metric
    :param metric_to_watch:             acc or loss
    :return: bool, True if new metric is better than current
    """
    return metric_to_watch == "acc" and new_metric > current_best_metric or (metric_to_watch == "loss" and current_best_metric > new_metric)


def makedirs_if_not_exists(dir_path: str):
    """
    make new directory in dir_path if it doesn't exists
        :param dir_path - full path of directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
