import os
import random
import cv2
import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List
from super_gradients.training.utils.utils import AverageMeter
from tqdm import tqdm

from super_gradients.training.utils.detection_utils import Anchors


class LocalLogger:
    def __init__(self):
        pass

    def add_images(self, tag: str, images: np.array):
        tag += '.jpg'
        cv2.imwrite(tag, images)

    def add_figure(self, tag: str, figure):
        path = str(os.getcwd()) + "/workspace/super-gradients/" + tag.replace(" ", "_") + '.jpg'
        # print(tag)
        figure.savefig(path)

    def add_text(self):
        pass


class DatasetStatistics:

    def __init__(self, data_loader: torch.utils.data.DataLoader,
                 classes_names: List[str],
                 summary_params: Dict = {},
                 title: str = 'MyData',
                 logger=LocalLogger(), anchors: Anchors=None):
        self.data_loader = data_loader
        self._classes_names = classes_names
        self._num_classes = len(classes_names)
        self._labels: np.array
        self.anchors = anchors
        self._summary_params = {'max_batches': 30,
                                'sample_images': 32,
                                'plot_sample_batch': True,
                                'plot_class_distribution': True,
                                'plot_object_size_distribution': True,
                                'plot_anchor_distribution': anchors is not None}
        self._summary_params.update(summary_params)
        # TODO: Get inside both loggers
        self._font_size = 15
        self.title = title

        self._logger = logger

    def analyze(self):
        self._get_dataset_metadata()

        if self._summary_params['plot_sample_batch']:
            self._get_sample_batch()
        if self._summary_params['plot_class_distribution']:
            self._analyze_class_distribution()
        if self._summary_params['plot_object_size_distribution']:
            self._analyze_object_size_distributions()
        if self._summary_params['plot_anchor_distribution']:
            self._analyze_anchors_coverage()

    def _get_sample_batch(self, ):
        images, labels = next(iter(self.data_loader))
        samples = images[:self._summary_params['sample_images']] if images.shape[0] > self._summary_params[
            'sample_images'] else images
        pred = [torch.zeros(size=(0, 6)) for _ in range(len(samples))]
        # TODO: Not working
        # result_images = DetectionVisualization.visualize_batch(image_tensor=samples,
        #                                                        pred_boxes=pred,
        #                                                        target_boxes=copy.deepcopy(labels),
        #                                                        batch_name=self.title,
        #                                                        class_names=self._classes_names,
        #                                                        box_thickness=1,
        #                                                        gt_alpha=1.0)
        # self._logger.add_images(tag=f'{self.title} sample images',
        #                         images=np.stack(result_images).transpose([0, 3, 1, 2])[:, ::-1, :, :])

    def _get_dataset_metadata(self):
        all_labels: List = []
        color_mean, color_std = AverageMeter(), AverageMeter()
        for i, (images, labels) in enumerate(tqdm(self.data_loader)):
            if i >= self._summary_params['max_batches'] > 0:
                break
            # print(labels)
            # print(images.shape)
            all_labels.append(labels)
            color_mean.update(torch.mean(images, dim=[0, 2, 3]), 1)
            color_std.update(torch.std(images, dim=[0, 2, 3]), 1)
            # TODO Should happen only once - only work on rectangular images?
            self._image_size = images[-1].numpy().shape[1]

        # TODO Check if working
        self._labels = torch.cat(all_labels, dim=0)[1:].numpy()

    def _analyze_class_distribution(self):
        hist, edges = np.histogram(self._labels[:, 0], self._num_classes)

        f = plt.figure(figsize=[10, 8])

        plt.bar(range(self._num_classes), hist, width=0.5, color='#0504aa', alpha=0.7)
        plt.xlim(-1, self._num_classes)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value', fontsize=self._font_size)
        plt.ylabel('Frequency', fontsize=self._font_size)
        plt.xticks(fontsize=self._font_size)
        plt.yticks(fontsize=self._font_size)
        plt.title(f'{self.title} class distribution', fontsize=self._font_size)

        self._logger.add_figure(f"{self.title} class distribution", figure=f)
        # text_dist = ''
        # for i, val in enumerate(hist):
        #     text_dist += f'[{i}]: {val}, '

        # self._logger.add_text(tag=f"{self.title} class distribution", text_string=text_dist)

    def _analyze_object_size_distributions(self):
        hist, xedges, yedges = np.histogram2d(self._labels[:, 4], self._labels[:, 3], 50)  # x and y are deliberately switched

        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(f'{self.title} boxes w/h distribution')
        ax = fig.add_subplot(121)
        ax.set_xlabel('W', fontsize=self._font_size)
        ax.set_ylabel('H', fontsize=self._font_size)
        plt.imshow(np.log(hist + 1), interpolation='nearest', origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

        # scatter plot
        if len(self._labels) > 10000:
            # we randomly sample just 10000 objects so that the scatter plot will not get too dense
            self._labels = self._labels[np.random.randint(0, len(self._labels) - 1, 10000)]
        ax = fig.add_subplot(122)
        ax.set_xlabel('W', fontsize=self._font_size)
        ax.set_ylabel('H', fontsize=self._font_size)

        plt.scatter(self._labels[:, 3], self._labels[:, 4], marker='.')

        self._logger.add_figure(tag=f'{self.title} boxes width height distribution', figure=fig)

    def _analyze_anchors_coverage(self):
        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f'{self.title} anchors coverage')

        # box style plot
        ax = fig.add_subplot(121)
        ax.set_xlabel('W', fontsize=self._font_size)
        ax.set_ylabel('H', fontsize=self._font_size)

        print(self._image_size)
        ax.set_xlim([0, self._image_size])
        ax.set_ylim([0, self._image_size])
        print(self.anchors.shape)

        anchors = np.array(self.anchors).reshape(-1, 2)

        for i in range(len(anchors)):
            rect = self._get_rect(anchors[i][0], anchors[i][1])
            rect.set_alpha(0.3)
            rect.set_facecolor([random.random(), random.random(), random.random(), 0.3])
            ax.add_patch(rect)

        # distance from anchor plot
        ax = fig.add_subplot(122)
        ax.set_xlabel('W', fontsize=self._font_size)
        ax.set_ylabel('H', fontsize=self._font_size)

        x = np.arange(1, self._image_size, 1)
        y = np.arange(1, self._image_size, 1)
        xx, yy = np.meshgrid(x, y, sparse=False)
        points = np.concatenate([xx.reshape(1, -1), yy.reshape(1, -1)])

        color = self._get_score(anchors, points, self._image_size)

        ax.set_xlabel('W', fontsize=self._font_size)
        ax.set_ylabel('H', fontsize=self._font_size)
        plt.imshow(color, interpolation='nearest', origin='lower',
                   extent=[0, self._image_size, 0, self._image_size])

        # calculate the coverage for the dataset labels
        cover_masks = []
        for i in range(len(anchors)):
            w_max = (anchors[i][0] / self._image_size) * 4
            w_min = (anchors[i][0] / self._image_size) * 0.25
            h_max = (anchors[i][1] / self._image_size) * 4
            h_min = (anchors[i][1] / self._image_size) * 0.25
            cover_masks.append(np.logical_and(
                np.logical_and(np.logical_and(self._labels[:, 3] < w_max, self._labels[:, 3] > w_min),
                               self._labels[:, 4] < h_max), self._labels[:, 4] > h_min))
        cover_masks = np.stack(cover_masks)
        coverage = np.count_nonzero(np.any(cover_masks, axis=0)) / len(self._labels)

        self._labels.add_figure(tag=f'{self.title} anchors coverage', figure=fig)

    # def _analyze_detection(self):
        # self.sg_logger.add_text(tag=f'{title} Statistics', text_string=summary)
        # self.sg_logger.flush()

        # except Exception as e:
        # any exception is caught here. we dont want the DatasetStatisticsLogger to crash any training
        # DatasetStatisticsTensorboardLogger.logger.error(f'dataset analysis failed: {e}')
        # print(e)
        # exit(0)

    @staticmethod
    def _get_rect(w, h):
        min_w = w / 4.0
        min_h = h / 4.0
        return Rectangle((min_w, min_h), w * 4 - min_w, h * 4 - min_h, linewidth=1, edgecolor='b', facecolor='none')

    @staticmethod
    def _get_score(anchors: np.ndarray, points: np.ndarray, image_size: int):
        ratio = anchors[:, :, None] / points[:, ]
        min_ratio = 1 - np.minimum(ratio, 1 / ratio)
        min_ratio = np.max(min_ratio, axis=1)
        to_closest_anchor = np.min(min_ratio, axis=0)
        to_closest_anchor[to_closest_anchor > 0.75] = 2
        return to_closest_anchor.reshape(image_size - 1, -1)
