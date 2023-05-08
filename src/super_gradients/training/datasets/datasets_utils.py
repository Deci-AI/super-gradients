import copy
import os
import random
import uuid
from abc import ABC, abstractmethod
from multiprocessing import Value, Lock
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib.patches import Rectangle
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, InterpolationMode, RandomResizedCrop
from tqdm import tqdm

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Callbacks, Transforms
from super_gradients.common.registry.registry import register_collate_function, register_callback, register_transform
from super_gradients.training.datasets.auto_augment import rand_augment_transform
from super_gradients.training.utils.detection_utils import DetectionVisualization, Anchors
from super_gradients.training.utils.distributed_training_utils import get_local_rank, get_world_size
from super_gradients.training.utils.utils import AverageMeter


def get_mean_and_std_torch(data_dir=None, dataloader=None, num_workers=4, RandomResizeSize=224):
    """
    A function for getting the mean and std of large datasets using pytorch dataloader and gpu functionality.

    :param data_dir: String, path to none-library dataset folder. For example "/data/Imagenette" or "/data/TinyImagenet"
    :param dataloader: a torch DataLoader, as it would feed the data into the trainer (including transforms etc).
    :param RandomResizeSize: Int, the size of the RandomResizeCrop as it appears in the DataInterface (for example, for Imagenet,
    this value should be 224).
    :return: 2 lists,mean and std, each one of len 3 (1 for each channel)
    """
    assert data_dir is None or dataloader is None, "Please provide either path to data folder or DataLoader, not both."

    if dataloader is None:
        traindir = os.path.join(os.path.abspath(data_dir), "train")
        trainset = ImageFolder(
            traindir, transforms.Compose([transforms.RandomResizedCrop(RandomResizeSize), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        )
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=num_workers)

    print(f"Calculating on {len(dataloader.dataset.targets)} Training Samples")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    h, w = 0, 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            print(f"Min: {inputs.min()}, Max: {inputs.max()}")
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
    mean = chsum / len(trainset) / h / w
    print(f"mean: {mean.view(-1)}")

    chsum = None
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(chsum / (len(trainset) * h * w - 1))
    print(f"std: {std.view(-1)}")
    return mean.view(-1).cpu().numpy().tolist(), std.view(-1).cpu().numpy().tolist()


class AbstractCollateFunction(ABC):
    """
    A collate function (for torch DataLoader)
    """

    @abstractmethod
    def __call__(self, batch):
        pass


@register_collate_function()
class ComposedCollateFunction(AbstractCollateFunction):
    """
    A function (for torch DataLoader) which executes a sequence of sub collate functions
    """

    def __init__(self, functions: list):
        self.functions = functions

    def __call__(self, batch):
        for f in self.functions:
            batch = f(batch)
        return batch


class AtomicInteger:
    def __init__(self, value: int = 0):
        self._value = Value("i", value)

    def __set__(self, instance, value):
        self._value.value = value

    def __get__(self, instance, owner):
        return self._value.value


@register_collate_function()
class MultiScaleCollateFunction(AbstractCollateFunction):
    """
    a collate function to implement multi-scale data augmentation
    according to https://arxiv.org/pdf/1612.08242.pdf
    """

    _counter = AtomicInteger(0)
    _current_size = AtomicInteger(0)
    _lock = Lock()

    def __init__(self, target_size: int = None, min_image_size: int = None, max_image_size: int = None, image_size_steps: int = 32, change_frequency: int = 10):
        """
        set parameters for the multi-scale collate function
        the possible image sizes are in range [min_image_size, max_image_size] in steps of image_size_steps
        a new size will be randomly selected every change_frequency calls to the collate_fn()
            :param target_size: scales will be [0.66 * target_size, 1.5 * target_size]
            :param min_image_size: the minimum size to scale down to (in pixels)
            :param max_image_size: the maximum size to scale up to (in pixels)
            :param image_size_steps: typically, the stride of the net, which defines the possible image
                    size multiplications
            :param change_frequency:
        """
        assert target_size is not None or (
            max_image_size is not None and min_image_size is not None
        ), "either target_size or min_image_size and max_image_size has to be set"
        assert target_size is None or max_image_size is None, "target_size and max_image_size cannot be both defined"

        if target_size is not None:
            min_image_size = int(0.66 * target_size - ((0.66 * target_size) % image_size_steps) + image_size_steps)
            max_image_size = int(1.5 * target_size - ((1.5 * target_size) % image_size_steps))

        print("Using multi-scale %g - %g" % (min_image_size, max_image_size))

        self.sizes = np.arange(min_image_size, max_image_size + image_size_steps, image_size_steps)
        self.image_size_steps = image_size_steps
        self.frequency = change_frequency
        self._current_size = random.choice(self.sizes)

    def __call__(self, batch):

        with self._lock:

            # Important: this implementation was tailored for a specific input. it assumes the batch is a tuple where
            # the images are the first item
            assert isinstance(batch, tuple), "this collate function expects the input to be a tuple (images, labels)"
            images = batch[0]
            if self._counter % self.frequency == 0:
                self._current_size = random.choice(self.sizes)
            self._counter += 1

            assert images.shape[2] % self.image_size_steps == 0 and images.shape[3] % self.image_size_steps == 0, (
                "images sized not divisible by %d. (resize images before calling multi_scale)" % self.image_size_steps
            )

            if self._current_size != max(images.shape[2:]):
                ratio = float(self._current_size) / max(images.shape[2:])
                new_size = (int(round(images.shape[2] * ratio)), int(round(images.shape[3] * ratio)))
                images = F.interpolate(images, size=new_size, mode="bilinear", align_corners=False)

            return images, batch[1]


class AbstractPrePredictionCallback(ABC):
    """
    Abstract class for forward pass preprocessing function, to be used by passing its inheritors through training_params
     pre_prediction_callback keyword arg.

    Should implement __call__ and return images, targets after applying the desired preprocessing.
    """

    @abstractmethod
    def __call__(self, inputs, targets, batch_idx):
        pass


class MultiscalePrePredictionCallback(AbstractPrePredictionCallback):
    """
    Mutiscale pre-prediction callback pass function.

    When passed through train_params images, targets will be applied by the below transform to support multi scaling
    on the fly.

    After each self.frequency forward passes, change size randomly from
     (input_size-self.multiscale_range*self.image_size_steps, input_size-(self.multiscale_range-1)*self.image_size_steps,
     ...input_size+self.multiscale_range*self.image_size_steps)


    :param multiscale_range: Range of values for resize sizes as discussed above (default=5)
    :param image_size_steps: Image step sizes as discussed abov (default=32)
    :param change_frequency: The frequency to apply change in input size.
    """

    def __init__(self, multiscale_range: int = 5, image_size_steps: int = 32, change_frequency: int = 10):

        self.multiscale_range = multiscale_range
        self.image_size_steps = image_size_steps
        self.frequency = change_frequency
        self.rank = None
        self.is_distributed = None
        self.sampled_imres_once = False
        self.new_input_size = None

    def __call__(self, inputs, targets, batch_idx):
        if self.rank is None:
            self.rank = get_local_rank()
        if self.is_distributed is None:
            self.is_distributed = get_world_size() > 1

        # GENERATE A NEW SIZE AND BROADCAST IT TO THE THE OTHER RANKS SO THEY HAVE THE SAME SCALE
        input_size = inputs.shape[2:]
        if batch_idx % self.frequency == 0:
            tensor = torch.LongTensor(2).to(inputs.device)

            if self.rank == 0:
                size_factor = input_size[1] * 1.0 / input_size[0]
                min_size = int(input_size[0] / self.image_size_steps) - self.multiscale_range
                max_size = int(input_size[0] / self.image_size_steps) + self.multiscale_range
                random_size = (min_size, max_size)
                if self.sampled_imres_once:
                    size = random.randint(*random_size)
                else:
                    # sample the biggest resolution first to make sure the run fits into the GPU memory
                    size = max_size
                    self.sampled_imres_once = True
                size = (int(self.image_size_steps * size), self.image_size_steps * int(size * size_factor))
                tensor[0] = size[0]
                tensor[1] = size[1]

            if self.is_distributed:
                dist.barrier()
                dist.broadcast(tensor, 0)

            self.new_input_size = (tensor[0].item(), tensor[1].item())

        scale_y = self.new_input_size[0] / input_size[0]
        scale_x = self.new_input_size[1] / input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = torch.nn.functional.interpolate(inputs, size=self.new_input_size, mode="bilinear", align_corners=False)
        return inputs, targets


@register_callback(Callbacks.DETECTION_MULTISCALE_PREPREDICTION)
class DetectionMultiscalePrePredictionCallback(MultiscalePrePredictionCallback):
    """
    Mutiscalepre-prediction callback for object detection.

    When passed through train_params images, targets will be applied by the below transform to support multi scaling
    on the fly.

    After each self.frequency forward passes, change size randomly from
     (input_size-self.multiscale_range*self.image_size_steps, input_size-(self.multiscale_range-1)*self.image_size_steps,
     ...input_size+self.multiscale_range*self.image_size_steps) and apply the same rescaling to the box coordinates.


    :param multiscale_range: Range of values for resize sizes as discussed above (default=5)
    :param image_size_steps: Image step sizes as discussed abov (default=32)
    :param change_frequency: The frequency to apply change in input size.

    """

    def __call__(self, inputs, targets, batch_idx):
        # RESCALE THE IMAGE FIRST WITH SUPER(), AND IF RESCALING HAS ACTUALLY BEEN DONE APPLY TO BOXES AS WELL
        input_size = inputs.shape[2:]
        inputs, targets = super(DetectionMultiscalePrePredictionCallback, self).__call__(inputs, targets, batch_idx)
        new_input_size = inputs.shape[2:]
        scale_y = new_input_size[0] / input_size[0]
        scale_x = new_input_size[1] / input_size[1]
        if scale_x != 1 or scale_y != 1:
            targets[..., 2::2] = targets[..., 2::2] * scale_x
            targets[..., 3::2] = targets[..., 3::2] * scale_y
        return inputs, targets


_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


def _pil_interp(method):
    if method == "bicubic":
        return InterpolationMode.BICUBIC
    elif method == "lanczos":
        return InterpolationMode.LANCZOS
    elif method == "hamming":
        return InterpolationMode.HAMMING
    elif method == "nearest":
        return InterpolationMode.NEAREST
    elif method == "bilinear":
        return InterpolationMode.BILINEAR
    elif method == "box":
        return InterpolationMode.BOX
    else:
        raise ValueError(
            "interpolation type must be one of ['bilinear', 'bicubic', 'lanczos', 'hamming', "
            "'nearest', 'box'] for explicit interpolation type, or 'random' for random"
        )


_RANDOM_INTERPOLATION = (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)


@register_transform(Transforms.RandomResizedCropAndInterpolation)
class RandomResizedCropAndInterpolation(RandomResizedCrop):
    """
    Crop the given PIL Image to random size and aspect ratio with explicitly chosen or random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    :param size: Expected output size of each edge
    :param scale: Range of size of the origin size cropped
    :param ratio: Range of aspect ratio of the origin aspect ratio cropped
    :param interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation="default"):
        super(RandomResizedCropAndInterpolation, self).__init__(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        elif interpolation == "default":
            self.interpolation = InterpolationMode.BILINEAR
        else:
            self.interpolation = _pil_interp(interpolation)

    def forward(self, img: Image) -> Image:
        """
        :param img: Image to be cropped and resized.
        :return: Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return torchvision.transforms.functional.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


STAT_LOGGER_FONT_SIZE = 15


class DatasetStatisticsTensorboardLogger:

    logger = get_logger(__name__)
    DEFAULT_SUMMARY_PARAMS = {
        "sample_images": 32,  # by default, 32 images will be sampled from each dataset
        "plot_class_distribution": True,
        "plot_box_size_distribution": True,
        "plot_anchors_coverage": True,
        "max_batches": 30,
    }

    def __init__(self, sg_logger, summary_params: dict = DEFAULT_SUMMARY_PARAMS):
        self.sg_logger = sg_logger
        self.summary_params = {**DatasetStatisticsTensorboardLogger.DEFAULT_SUMMARY_PARAMS, **summary_params}

    def analyze(self, data_loader: torch.utils.data.DataLoader, title: str, all_classes: List[str], anchors: list = None):
        """
        :param data_loader: the dataset data loader
        :param dataset_params: the dataset parameters
        :param title: the title for this dataset (i.e. Coco 2017 test set)
        :param anchors: the list of anchors used by the model. applicable only for detection datasets
        :param all_classes: the list of all classes names
        """
        # FIXME: UNCOMMENT AND APPLY TO NEW DetectionDataSet ONCE ITS MERGED
        # if isinstance(data_loader.dataset, DetectionDataSet):
        #     self._analyze_detection(data_loader=data_loader, title=title,
        #                             all_classes=all_classes, anchors=anchors)
        # else:
        #     DatasetStatisticsTensorboardLogger.logger.warning('only DetectionDataSet are currently supported')
        DatasetStatisticsTensorboardLogger.logger.warning("only DetectionDataSet are currently supported")

    def _analyze_detection(self, data_loader, title, all_classes, anchors=None):
        """
        Analyze a detection dataset

        :param data_loader: the dataset data loader
        :param dataset_params: the dataset parameters
        :param all_classes: the list of all classes names
        :param title: the title for this dataset (i.e. Coco 2017 test set)
        :param anchors: the list of anchors used by the model. if not provided, anchors coverage will not be analyzed
        """
        try:
            color_mean = AverageMeter()
            color_std = AverageMeter()
            all_labels = []
            image_size = 0
            for i, (images, labels) in enumerate(tqdm(data_loader)):

                if i >= self.summary_params["max_batches"] > 0:
                    break

                if i == 0:
                    image_size = max(images[0].shape[1], images[0].shape[2])
                    if images.shape[0] > self.summary_params["sample_images"]:
                        samples = images[: self.summary_params["sample_images"]]
                    else:
                        samples = images

                    pred = [torch.zeros(size=(0, 6)) for _ in range(len(samples))]
                    try:
                        result_images = DetectionVisualization.visualize_batch(
                            image_tensor=samples,
                            pred_boxes=pred,
                            target_boxes=copy.deepcopy(labels),
                            batch_name=title,
                            class_names=all_classes,
                            box_thickness=1,
                            gt_alpha=1.0,
                        )

                        self.sg_logger.add_images(tag=f"{title} sample images", images=np.stack(result_images).transpose([0, 3, 1, 2])[:, ::-1, :, :])
                    except Exception as e:
                        DatasetStatisticsTensorboardLogger.logger.error(f"Dataset Statistics failed at adding an example batch:\n{e}")
                        return

                all_labels.append(labels)
                color_mean.update(torch.mean(images, dim=[0, 2, 3]), 1)
                color_std.update(torch.std(images, dim=[0, 2, 3]), 1)

            all_labels = torch.cat(all_labels, dim=0)[1:].numpy()

            try:
                if self.summary_params["plot_class_distribution"]:
                    self._analyze_class_distribution(labels=all_labels, num_classes=len(all_classes), title=title)
            except Exception as e:
                DatasetStatisticsTensorboardLogger.logger.error(f"Dataset Statistics failed at analyzing class distributions.\n{e}")
                return

            try:
                if self.summary_params["plot_box_size_distribution"]:
                    self._analyze_object_size_distribution(labels=all_labels, title=title)
            except Exception as e:
                DatasetStatisticsTensorboardLogger.logger.error(f"Dataset Statistics failed at analyzing object size " f"distributions.\n{e}")
                return

            summary = ""
            summary += f"dataset size: {len(data_loader)}  \n"
            summary += f"color mean: {color_mean.average}  \n"
            summary += f"color std: {color_std.average}  \n"

            try:
                if anchors is not None and image_size > 0:
                    coverage = self._analyze_anchors_coverage(anchors=anchors, image_size=image_size, title=title, labels=all_labels)
                    summary += f"anchors: {anchors}  \n"
                    summary += f"anchors coverage: {coverage}  \n"
            except Exception as e:
                DatasetStatisticsTensorboardLogger.logger.error(f"Dataset Statistics failed at analyzing anchors " f"coverage.\n{e}")
                return

            self.sg_logger.add_text(tag=f"{title} Statistics", text_string=summary)
            self.sg_logger.flush()

        except Exception as e:
            DatasetStatisticsTensorboardLogger.logger.error(f"dataset analysis failed!\n{e}")

    def _analyze_class_distribution(self, labels: list, num_classes: int, title: str):
        hist, edges = np.histogram(labels[:, 0], num_classes)

        f = plt.figure(figsize=[10, 8])

        plt.bar(range(num_classes), hist, width=0.5, color="#0504aa", alpha=0.7)
        plt.xlim(-1, num_classes)
        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Value", fontsize=STAT_LOGGER_FONT_SIZE)
        plt.ylabel("Frequency", fontsize=STAT_LOGGER_FONT_SIZE)
        plt.xticks(fontsize=STAT_LOGGER_FONT_SIZE)
        plt.yticks(fontsize=STAT_LOGGER_FONT_SIZE)
        plt.title(f"{title} class distribution", fontsize=STAT_LOGGER_FONT_SIZE)

        self.sg_logger.add_figure(f"{title} class distribution", figure=f)
        text_dist = ""
        for i, val in enumerate(hist):
            text_dist += f"[{i}]: {val}, "

        self.sg_logger.add_text(tag=f"{title} class distribution", text_string=text_dist)

    def _analyze_object_size_distribution(self, labels: list, title: str):
        """
        This function will add two plots to the tensorboard.
        one is a 2D histogram and the other is a scatter plot. in both cases the X axis is the object width and Y axis
        is the object width (both normalized by image size)
        :param labels: all the labels of the dataset of the shape [class_label, x_center, y_center, w, h]
        :param title: the dataset title
        """

        # histogram plot
        hist, xedges, yedges = np.histogram2d(labels[:, 4], labels[:, 3], 50)  # x and y are deliberately switched

        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(f"{title} boxes w/h distribution")
        ax = fig.add_subplot(121)
        ax.set_xlabel("W", fontsize=STAT_LOGGER_FONT_SIZE)
        ax.set_ylabel("H", fontsize=STAT_LOGGER_FONT_SIZE)
        plt.imshow(np.log(hist + 1), interpolation="nearest", origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

        # scatter plot
        if len(labels) > 10000:
            # we randomly sample just 10000 objects so that the scatter plot will not get too dense
            labels = labels[np.random.randint(0, len(labels) - 1, 10000)]
        ax = fig.add_subplot(122)
        ax.set_xlabel("W", fontsize=STAT_LOGGER_FONT_SIZE)
        ax.set_ylabel("H", fontsize=STAT_LOGGER_FONT_SIZE)

        plt.scatter(labels[:, 3], labels[:, 4], marker=".")

        self.sg_logger.add_figure(tag=f"{title} boxes w/h distribution", figure=fig)

    @staticmethod
    def _get_rect(w, h):
        min_w = w / 4.0
        min_h = h / 4.0
        return Rectangle((min_w, min_h), w * 4 - min_w, h * 4 - min_h, linewidth=1, edgecolor="b", facecolor="none")

    @staticmethod
    def _get_score(anchors: np.ndarray, points: np.ndarray, image_size: int):
        """
        Calculate the ratio (and 1/ratio) between each anchor width and height and each point (representing a possible
        object width and height).
        i.e. for an anchor with w=10,h=20 the point w=11,h=25 will have the ratios 11/10=1.1 and 25/20=1.25
        or 10/11=0.91 and 20/25=0.8 respectively

        :param anchors: array of anchors of the shape [2,N]
        :param points: array of points of the shape [2,M]
        :param image_size the size of the input image

        :returns: an array of size [image_size - 1, image_size - 1] where each cell i,j represent the minimum ratio
        for that cell (point) from all anchors
        """

        ratio = (
            anchors[:, :, None]
            / points[
                :,
            ]
        )
        inv_ratio = 1 / ratio
        min_ratio = 1 - np.minimum(ratio, inv_ratio)
        min_ratio = np.max(min_ratio, axis=1)
        to_closest_anchor = np.min(min_ratio, axis=0)
        to_closest_anchor[to_closest_anchor > 0.75] = 2
        return to_closest_anchor.reshape(image_size - 1, -1)

    def _analyze_anchors_coverage(self, anchors: Anchors, image_size: int, labels: list, title: str):
        """
        This function will add anchors coverage plots to the tensorboard.
        :param anchors: a list of anchors
        :param image_size: the input image size for this training
        :param labels: all the labels of the dataset of the shape [class_label, x_center, y_center, w, h]
        :param title: the dataset title
        """

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f"{title} anchors coverage")

        # box style plot
        ax = fig.add_subplot(121)
        ax.set_xlabel("W", fontsize=STAT_LOGGER_FONT_SIZE)
        ax.set_ylabel("H", fontsize=STAT_LOGGER_FONT_SIZE)
        ax.set_xlim([0, image_size])
        ax.set_ylim([0, image_size])

        anchors_boxes = anchors.anchors.cpu().numpy()
        anchors_len = anchors.num_anchors

        anchors_boxes = anchors_boxes.reshape(-1, 2)

        for i in range(anchors_len):
            rect = self._get_rect(anchors_boxes[i][0], anchors_boxes[i][1])
            rect.set_alpha(0.3)
            rect.set_facecolor([random.random(), random.random(), random.random(), 0.3])
            ax.add_patch(rect)

        # distance from anchor plot
        ax = fig.add_subplot(122)
        ax.set_xlabel("W", fontsize=STAT_LOGGER_FONT_SIZE)
        ax.set_ylabel("H", fontsize=STAT_LOGGER_FONT_SIZE)

        x = np.arange(1, image_size, 1)
        y = np.arange(1, image_size, 1)

        xx, yy = np.meshgrid(x, y, sparse=False, indexing="xy")
        points = np.concatenate([xx.reshape(1, -1), yy.reshape(1, -1)])

        color = self._get_score(anchors_boxes, points, image_size)

        ax.set_xlabel("W", fontsize=STAT_LOGGER_FONT_SIZE)
        ax.set_ylabel("H", fontsize=STAT_LOGGER_FONT_SIZE)
        plt.imshow(color, interpolation="nearest", origin="lower", extent=[0, image_size, 0, image_size])

        # calculate the coverage for the dataset labels
        cover_masks = []
        for i in range(anchors_len):
            w_max = (anchors_boxes[i][0] / image_size) * 4
            w_min = (anchors_boxes[i][0] / image_size) * 0.25
            h_max = (anchors_boxes[i][1] / image_size) * 4
            h_min = (anchors_boxes[i][1] / image_size) * 0.25
            cover_masks.append(
                np.logical_and(np.logical_and(np.logical_and(labels[:, 3] < w_max, labels[:, 3] > w_min), labels[:, 4] < h_max), labels[:, 4] > h_min)
            )
        cover_masks = np.stack(cover_masks)
        coverage = np.count_nonzero(np.any(cover_masks, axis=0)) / len(labels)

        self.sg_logger.add_figure(tag=f"{title} anchors coverage", figure=fig)
        return coverage


def get_color_augmentation(rand_augment_config_string: str, color_jitter: tuple, crop_size=224, img_mean=[0.485, 0.456, 0.406]):
    """
    Returns color augmentation class. As these augmentation cannot work on top one another, only one is returned
    according to rand_augment_config_string

    :param rand_augment_config_string: string which defines the auto augment configurations.
                                       If none, color jitter will be returned. For possibile values see auto_augment.py
    :param color_jitter: tuple for color jitter value.
    :param crop_size: relevant only for auto augment
    :param img_mean: relevant only for auto augment
    :return: RandAugment transform or ColorJitter
    """
    if rand_augment_config_string:
        color_augmentation = rand_augment_transform(rand_augment_config_string, crop_size, img_mean)

    else:  # RandAugment includes colorjitter like augmentations, both cannot be applied together.
        color_augmentation = transforms.ColorJitter(*color_jitter)
    return color_augmentation


def worker_init_reset_seed(worker_id):
    """
    Make sure each process has different random seed, especially for 'fork' method.
    Check https://github.com/pytorch/pytorch/issues/63311 for more details.

    :param worker_id: placeholder (needs to be passed to DataLoader init).
    """
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
