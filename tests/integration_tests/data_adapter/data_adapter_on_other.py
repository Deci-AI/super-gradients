import cv2
import torch
from torch.utils.data import DataLoader

from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from data_gradients.datasets.detection.yolo_format_detection_dataset import YoloFormatDetectionDataset
from data_gradients.datasets.segmentation.voc_segmentation_dataset import VOCSegmentationDataset

from super_gradients.training.utils.collate_fn import (
    DetectionDataloaderAdapter,
    SegmentationDataloaderAdapter,
    ClassificationDataloaderAdapter,
    DetectionDatasetAdapterCollateFN,
    SegmentationDatasetAdapterCollateFN,
)


def datagradients_detection():
    data_dir = "/Users/Louis.Dupont/Desktop/DataGradients/data-examples/solar_panels"  # ...  # Fill with the <path-to>/data_dir
    class_names = ["Cell", "Cell-Multi", "No-Anomaly", "Shadowing", "Unclassified"]  # Fill with the list of class names

    train_set = YoloFormatDetectionDataset(root_dir=data_dir, images_dir="train/images", labels_dir="train/labels")
    val_set = YoloFormatDetectionDataset(root_dir=data_dir, images_dir="valid/images", labels_dir="valid/labels")

    analyzer = DetectionAnalysisManager(
        report_title="coco2017_val",
        train_data=train_set,
        val_data=val_set,
        class_names=class_names,
        batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
        is_label_first=True,
        bbox_format="cxcywh",
    )
    analyzer.run()

    train_loader = DataLoader(
        train_set,
        batch_size=20,
        num_workers=0,
        collate_fn=DetectionDatasetAdapterCollateFN(adapter_cache_path=analyzer.data_config.cache_path, n_classes=80),
        drop_last=True,
    )
    DetectionDatasetAdapterCollateFN.setup_adapter(train_loader)

    val_loader = DataLoader(
        val_set,
        batch_size=20,
        num_workers=0,
        collate_fn=DetectionDatasetAdapterCollateFN(adapter_cache_path=analyzer.data_config.cache_path, n_classes=80),
        drop_last=True,
    )
    DetectionDatasetAdapterCollateFN.setup_adapter(val_loader)

    for images, labels in train_loader:
        assert images.ndim == 4
        assert images.shape[:2] == torch.Size([20, 3])
        assert labels.ndim == 2
        assert labels.shape[-1] == 6

    for images, labels in val_loader:
        assert images.ndim == 4
        assert images.shape[:2] == torch.Size([20, 3])
        assert labels.ndim == 2
        assert labels.shape[-1] == 6


def datagradients_segmentation():
    data_dir = "/Users/Louis.Dupont/Desktop/DataGradients/data-examples/VOCdevkit"  # ...  # Fill with the <path-to>/data_dir

    train_set = VOCSegmentationDataset(root_dir=data_dir, split="train", download=False, year=2007)
    val_set = VOCSegmentationDataset(root_dir=data_dir, split="val", download=False, year=2007)

    class ResizedVOC:
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            image, label = self.dataset[index]
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
            return image, label

        def __len__(self):
            return len(self.dataset)

    analyzer = SegmentationAnalysisManager(
        report_title="coco2017_val",
        train_data=ResizedVOC(dataset=train_set),
        val_data=ResizedVOC(dataset=val_set),
        class_names=train_set.class_names,
        batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
    )
    analyzer.run()

    train_loader = DataLoader(
        ResizedVOC(train_set),
        batch_size=20,
        num_workers=0,
        collate_fn=SegmentationDatasetAdapterCollateFN(adapter_cache_path=analyzer.data_config.cache_path, n_classes=len(train_set.class_names)),
        drop_last=True,
    )
    SegmentationDatasetAdapterCollateFN.setup_adapter(train_loader)

    val_loader = DataLoader(
        ResizedVOC(val_set),
        batch_size=20,
        num_workers=0,
        collate_fn=SegmentationDatasetAdapterCollateFN(adapter_cache_path=analyzer.data_config.cache_path, n_classes=len(val_set.class_names)),
        drop_last=True,
    )
    SegmentationDatasetAdapterCollateFN.setup_adapter(val_loader)

    for images, labels in train_loader:
        assert images.shape == torch.Size([20, 3, 512, 512])
        assert labels.shape == torch.Size([20, 512, 512])

    for images, labels in val_loader:
        assert images.shape == torch.Size([20, 3, 512, 512])
        assert labels.shape == torch.Size([20, 512, 512])


def torch_classification():
    from torchvision.datasets.caltech import Caltech101
    from torchvision.transforms import Compose, ToTensor, Resize

    from data_gradients.managers.classification_manager import ClassificationAnalysisManager

    class ToRGB:
        def __call__(self, pic):
            return pic.convert("RGB")

    train_set = Caltech101(root="./data", download=True, transform=Compose([ToRGB(), ToTensor(), Resize((512, 512))]))

    analyzer = ClassificationAnalysisManager(
        train_data=train_set,
        val_data=train_set,
        report_title="Caltech101",
        class_names=train_set.categories,
        batches_early_stop=4,
        n_image_channels=3,
        use_cache=True,
    )
    analyzer.run()

    train_loader = ClassificationDataloaderAdapter.from_dataset(
        dataset=train_set,
        adapter_cache_path=analyzer.data_config.cache_path,
        batch_size=20,
    )

    images, labels = next(iter(train_loader))
    assert images.shape == torch.Size([20, 3, 512, 512])
    assert labels.shape == torch.Size([20])


def torchvision_detection():
    # data_dir = "/Users/Louis.Dupont/Desktop/DataGradients/data-examples/solar_panels"  # ...  # Fill with the <path-to>/data_dir
    # class_names = ["Cell", "Cell-Multi", "No-Anomaly", "Shadowing", "Unclassified"]  # Fill with the list of class names

    from torchvision.datasets import VOCDetection

    data_dir = "/Users/Louis.Dupont/Desktop/DataGradients/data-examples/VOCdevkit/"  # ...  # Fill with the <path-to>/data_dir

    from torchvision.transforms import Resize, Compose

    train_set = VOCDetection(root=data_dir, image_set="train", download=False, year="2007", transform=Compose([Resize(size=(720, 720))]))
    val_set = VOCDetection(root=data_dir, image_set="val", download=False, year="2007", transform=Compose([Resize(size=(720, 720))]))

    import numpy as np

    PASCAL_VOC_CLASS_NAMES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def voc_format_to_bbox(sample: tuple) -> np.ndarray:
        target_annotations = sample[1]
        targets = []
        for target in target_annotations["annotation"]["object"]:
            target_bbox = target["bndbox"]
            target_np = np.array(
                [
                    PASCAL_VOC_CLASS_NAMES.index(target["name"]),
                    float(target_bbox["xmin"]),
                    float(target_bbox["ymin"]),
                    float(target_bbox["xmax"]),
                    float(target_bbox["ymax"]),
                ]
            )
            targets.append(target_np)
        return np.array(targets, dtype=float)

    from data_gradients.dataset_adapters.config.data_config import DetectionDataConfig

    analyzer = DetectionAnalysisManager(
        report_title="VOC_from_torch",
        train_data=train_set,
        val_data=val_set,
        labels_extractor=voc_format_to_bbox,
        class_names=PASCAL_VOC_CLASS_NAMES,
        # class_names=train_set,
        batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
        is_label_first=True,
        bbox_format="cxcywh",
    )

    analyzer.run()
    adapter_config = DetectionDataConfig(labels_extractor=voc_format_to_bbox, cache_path=analyzer.data_config.cache_path)
    train_loader = DetectionDataloaderAdapter.from_dataset(
        dataset=train_set,
        adapter_config=adapter_config,
        batch_size=20,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DetectionDataloaderAdapter.from_dataset(
        dataset=train_set,
        adapter_config=adapter_config,
        batch_size=20,
        num_workers=0,
        drop_last=True,
    )

    for images, labels in train_loader:
        assert images.ndim == 4
        assert images.shape[:2] == torch.Size([20, 3])
        assert labels.ndim == 2
        assert labels.shape[-1] == 6

    for images, labels in val_loader:
        assert images.ndim == 4
        assert images.shape[:2] == torch.Size([20, 3])
        assert labels.ndim == 2
        assert labels.shape[-1] == 6


def torchvision_segmentation():
    # data_dir = "/Users/Louis.Dupont/Desktop/DataGradients/data-examples/solar_panels"  # ...  # Fill with the <path-to>/data_dir
    # class_names = ["Cell", "Cell-Multi", "No-Anomaly", "Shadowing", "Unclassified"]  # Fill with the list of class names

    from torchvision.datasets import VOCSegmentation

    data_dir = "/Users/Louis.Dupont/Desktop/DataGradients/data-examples/VOCdevkit/"  # ...  # Fill with the <path-to>/data_dir

    from torchvision.transforms import Resize, Compose, InterpolationMode

    train_set = VOCSegmentation(
        root=data_dir,
        image_set="train",
        download=False,
        year="2007",
        transform=Compose([Resize(size=(720, 720))]),
        target_transform=Compose([Resize((720, 720), interpolation=InterpolationMode.NEAREST)]),
    )
    val_set = VOCSegmentation(
        root=data_dir,
        image_set="val",
        download=False,
        year="2007",
        transform=Compose([Resize(size=(720, 720))]),
        target_transform=Compose([Resize((720, 720), interpolation=InterpolationMode.NEAREST)]),
    )

    from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig

    analyzer = SegmentationAnalysisManager(
        report_title="VOC_SEG_from_torch2",
        train_data=train_set,
        val_data=val_set,
        class_names=list(range(256)),
        # class_names=train_set,
        batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
    )

    analyzer.run()
    adapter_config = SegmentationDataConfig(cache_path=analyzer.data_config.cache_path)
    train_loader = SegmentationDataloaderAdapter.from_dataset(
        dataset=train_set,
        adapter_config=adapter_config,
        batch_size=20,
        num_workers=0,
        drop_last=True,
    )
    val_loader = SegmentationDataloaderAdapter.from_dataset(
        dataset=train_set,
        adapter_config=adapter_config,
        batch_size=20,
        num_workers=0,
        drop_last=True,
    )

    for images, labels in train_loader:
        assert images.shape == torch.Size([20, 3, 720, 720])
        assert labels.shape == torch.Size([20, 720, 720])

    for images, labels in val_loader:
        assert images.shape == torch.Size([20, 3, 720, 720])
        assert labels.shape == torch.Size([20, 720, 720])


torch_classification()
# torchvision_segmentation()
# torchvision_detection()
# datagradients_detection()
# datagradients_segmentation()
# torch_classification()
