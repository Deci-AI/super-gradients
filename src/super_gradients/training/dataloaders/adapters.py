from typing import Iterable, Tuple
from torchvision.transforms import ToTensor
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter as DetectionDatasetAdapter
from data_gradients.dataset_adapters.segmentation_adapter import SegmentationDatasetAdapter as SegmentationDatasetAdapter
from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter as ClassificationDatasetAdapter
from super_gradients.common.registry.registry import register_dataset_adapters
import torch


@register_dataset_adapters()
class SGDetectionDatasetAdapter(DetectionDatasetAdapter):
    def adapt_batch(self, data) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        images, targets = super().adapt_batch(data)
        return images, targets


@register_dataset_adapters()
class SGSegmentationDatasetAdapter(SegmentationDatasetAdapter):
    def adapt_batch(self, data) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        images, targets = super().adapt_batch(data)
        return images, targets


@register_dataset_adapters()
class SGClassificationDatasetAdapter(ClassificationDatasetAdapter):
    def adapt_batch(self, data) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        images, targets = super().adapt_batch(data)
        images = ToTensor()(images)
        return images, targets
