import os
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataSet
import glob

PASCAL_VOC_2012_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor'
]


class PascalVOCDetectionDataSet(DetectionDataSet):
    """
    PascalVOC2012DtectionDataSet - Detection Data Set Class pascal_voc Data Set
    """

    def __init__(self, samples_sub_directory, targets_sub_directory, *args, **kwargs):
        self.samples_sub_directory = samples_sub_directory
        self.targets_sub_directory = targets_sub_directory

        super().__init__(*args, **kwargs)
        self.classes = PASCAL_VOC_2012_CLASSES

    def _generate_samples_and_targets(self):
        """
        _generate_samples_and_targets from subdirectories and then continue
        with super._generate_samples_and_targets

        """
        num_missing_files = 0
        self.img_files = glob.glob(self.root + self.samples_sub_directory + "*.jpg")
        num_files = len(self.img_files)
        for img_file in self.img_files:
            label_file = img_file.replace("images", "labels").replace(".jpg", ".txt")
            if os.path.exists(label_file):
                self.samples_targets_tuples_list.append((img_file, label_file))
            else:
                num_missing_files += 1
        if num_missing_files > 0:
            print(f'[WARNING] out of {num_files} lines, {num_missing_files} files were not loaded')

        super()._generate_samples_and_targets()
