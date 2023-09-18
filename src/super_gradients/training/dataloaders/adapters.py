from typing import Iterable, Tuple
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter as DetectionDatasetAdapter
from data_gradients.dataset_adapters.segmentation_adapter import SegmentationDatasetAdapter as SegmentationDatasetAdapter
from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter as ClassificationDatasetAdapter
from super_gradients.common.registry.registry import register_dataset_adapters
from super_gradients.training.utils.detection_utils import xyxy2cxcywh
import torch


@register_dataset_adapters()
class SGDetectionDatasetAdapter(DetectionDatasetAdapter):
    def adapt(self, data) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        images, targets = super().adapt(data)
        images = images / 255

        targets[:, :, 1:] = xyxy2cxcywh(targets[:, :, 1:])
        if not self.data_config.is_batch:
            images, targets = images[0], targets[0]
        return images, targets

    def adapt_sample(self, data) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        pass


@register_dataset_adapters()
class SGSegmentationDatasetAdapter(SegmentationDatasetAdapter):
    def adapt(self, data) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        images, targets = super().adapt(data)
        images = images / 255
        targets = targets.argmax(1)
        if not self.data_config.is_batch:
            images, targets = images[0], targets[0]
        return images, targets


@register_dataset_adapters()
class SGClassificationDatasetAdapter(ClassificationDatasetAdapter):
    def adapt(self, data) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        images, targets = super().adapt(data)
        images = images / 255
        if not self.data_config.is_batch:
            images, targets = images[0], targets[0]
        return images, targets
