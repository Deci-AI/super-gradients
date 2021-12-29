import os
import numpy as np
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataSet
from super_gradients.training.utils.detection_utils import convert_xyxy_bbox_to_xywh
import xml.etree.ElementTree as Et

PASCAL_VOC_2012_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor'
]


class PascalVOC2012DetectionDataSet(DetectionDataSet):
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
        # GENERATE SAMPLES AND TARGETS HERE SPECIFICALLY FOR PASCAL VOC 2012
        with open(self.root + os.path.sep + self.list_file_path, "r", encoding="utf-8") as lines:
            num_missing_files = 0
            num_files = 0
            for line in lines:
                num_files += 1
                image_path = os.path.join(self.root, self.samples_sub_directory, line.rstrip('\n') + '.jpg')
                labels_path = os.path.join(self.root, self.targets_sub_directory, line.rstrip('\n') + '.xml')

                if os.path.exists(labels_path) and os.path.exists(image_path):
                    self.samples_targets_tuples_list.append((image_path, labels_path))
                else:
                    num_missing_files += 1
            if num_missing_files > 0:
                print(f'[WARNING] out of {num_files} lines, {num_missing_files} files were not loaded')
        super()._generate_samples_and_targets()

    def target_loader(self, target_path: str) -> np.array:
        """
        target_loader -               load targets from xml file

            :param target_path:       xml file target path
            :return: np.array         parsed labels
        """
        target = None
        if os.path.isfile(target_path):
            try:
                target = self.parse_xml_file(target_path)
                if target.shape[0]:
                    assert target.shape[1] == 5, '> 5 label columns: %s' % target_path
                    assert (target >= 0).all(), 'negative labels: %s' % target_path
                    assert (target[:,
                            1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % target_path

            except Exception as ex:
                print(f'[WARNING] Caught Exception: {ex}, when trying to open {target_path}')

        return target

    def parse_xml_file(self, path):
        # labels_data = []
        with open(path, "r") as xml:
            tree = Et.parse(xml)
            root = tree.getroot()
            # GET ATTRIBUTES
            xml_size = root.find("size")
            width = int(xml_size.find("width").text)
            height = int(xml_size.find("height").text)
            objects = root.findall("object")
            labels = self._xyxy_to_normalized_xywh(objects, width, height)
        return labels

    def _xyxy_to_normalized_xywh(self, objects, width, height):
        labels = np.zeros((len(objects), 5))
        for i, _object in enumerate(objects):
            labels[0, 0] = (PASCAL_VOC_2012_CLASSES.index(_object.find("name").text))
            xml_bndbox = _object.find("bndbox")
            labels[i, 1] = int(xml_bndbox.find("xmin").text) / width
            labels[i, 2] = int(xml_bndbox.find("ymin").text) / height
            labels[i, 3] = int(xml_bndbox.find("xmax").text) / width
            labels[i, 4] = int(xml_bndbox.find("ymax").text) / height
        labels[:, 1:5] = convert_xyxy_bbox_to_xywh(labels[:, 1:5])
        return labels
