import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transform
from super_gradients.training.utils.segmentation_utils import RandomFlip, CropImageAndMask, PadShortToCropSize,\
    RandomRescale, Rescale

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as pycocotools_mask
except ModuleNotFoundError as ex:
    print("[WARNING]" + str(ex))

from super_gradients.training.datasets.datasets_conf import COCO_DEFAULT_CLASSES_TUPLES_LIST
from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet


class EmptyCoCoClassesSelectionException(Exception):
    pass


class CoCoSegmentationDataSet(SegmentationDataSet):
    """
    CoCoSegmentationDataSet - Segmentation Data Set Class for COCO 2017 Segmentation Data Set
    """

    def __init__(self, dataset_classes_inclusion_tuples_list: list = None, *args, **kwargs):
        # THERE ARE 91 CLASSES, INCLUDING BACKGROUND - BUT WE ENABLE THE USAGE OF SUBCLASSES, TO PARTIALLY USE THE DATA
        self.dataset_classes_inclusion_tuples_list = dataset_classes_inclusion_tuples_list or COCO_DEFAULT_CLASSES_TUPLES_LIST

        # OVERRIDE DEFAULT AUGMENTATIONS, IMG_SIZE, CROP SIZE
        dataset_hyper_params = kwargs['dataset_hyper_params']
        kwargs['img_size'] = dataset_hyper_params['img_size'] if 'img_size' in dataset_hyper_params.keys() else 608
        kwargs['crop_size'] = dataset_hyper_params['crop_size'] if 'crop_size' in dataset_hyper_params.keys() else 512
        # FIXME - Rescale before RandomRescale is kept for legacy support, consider removing it like most implementation
        #  papers regimes.
        kwargs["image_mask_transforms_aug"] = transform.Compose([RandomFlip(),
                                                                 Rescale(long_size=kwargs["img_size"]),
                                                                 RandomRescale(scales=(0.5, 2.0)),
                                                                 PadShortToCropSize(crop_size=kwargs['crop_size']),
                                                                 CropImageAndMask(crop_size=kwargs['crop_size'], mode="random")])
        super().__init__(*args, **kwargs)

        _, class_names = zip(*self.dataset_classes_inclusion_tuples_list)
        self.classes = class_names

    def _generate_samples_and_targets(self):
        """
        _generate_samples_and_targets
        """
        # FIRST OF ALL LOAD ALL OF THE ANNOTATIONS, AND CREATE THE PATH FOR THE PRE-PROCESSED MASKS
        self.annotations_file_path = os.path.join(self.root, self.targets_sub_directory, self.list_file_path)
        self.coco = COCO(self.annotations_file_path)

        # USE SUB-CLASSES OF THE ENTIRE COCO DATA SET, INSTEAD ALL OF THE DATA -> HIGHLY RELEVANT FOR TRANSFER LEARNING
        sub_dataset_image_ids_file_path = self.annotations_file_path.replace('json', 'pth')

        if os.path.exists(sub_dataset_image_ids_file_path):
            self.relevant_image_ids = torch.load(sub_dataset_image_ids_file_path)
        else:
            self.relevant_image_ids = self._sub_dataset_creation(sub_dataset_image_ids_file_path)

        for relevant_image_id in self.relevant_image_ids:
            img_metadata = self.coco.loadImgs(relevant_image_id)[0]
            image_path = os.path.join(self.root, self.samples_sub_directory, img_metadata['file_name'])
            mask_metadata_tuple = (relevant_image_id, img_metadata['height'], img_metadata['width'])
            self.samples_targets_tuples_list.append((image_path, mask_metadata_tuple))

    def target_loader(self, mask_metadata_tuple) -> Image:
        """
        target_loader
            :param mask_metadata_tuple:  A tuple of (coco_image_id, original_image_height, original_image_width)
            :return:                     The mask image created from the array
        """
        coco_image_id, original_image_h, original_image_w = mask_metadata_tuple
        coco_annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=coco_image_id))

        mask = self._generate_coco_segmentation_mask(coco_annotations, original_image_h, original_image_w)
        return Image.fromarray(mask)

    def _generate_coco_segmentation_mask(self, target_coco_annotations, h, w):
        """
        _generate_segmentation_mask - Extracts a segmentation mask
            :param target_coco_annotations:
            :param h:
            :param w:
            :return:
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        for i, instance in enumerate(target_coco_annotations):

            rle = pycocotools_mask.frPyObjects(instance['segmentation'], h, w)
            coco_segementation_mask = pycocotools_mask.decode(rle)

            if not self.dataset_classes_inclusion_tuples_list:
                # NO CLASSES WERE SELECTED FROM COCO'S 91 CLASSES - ERROR
                raise EmptyCoCoClassesSelectionException
            else:
                # FILTER OUT ALL OF THE MASKS OF INSTANCES THAT ARE NOT IN THE SUB-DATASET CLASSES
                class_category = instance['category_id']

                sub_classes_category_ids, _ = map(list, zip(*self.dataset_classes_inclusion_tuples_list))
                if class_category not in sub_classes_category_ids:
                    continue

                class_index = sub_classes_category_ids.index(class_category)
                if len(coco_segementation_mask.shape) < 3:
                    mask[:, :] += (mask == 0) * (coco_segementation_mask * class_index)
                else:
                    mask[:, :] += (mask == 0) * (((np.sum(coco_segementation_mask, axis=2)) > 0) * class_index).astype(
                        np.uint8)

        return mask

    def _sub_dataset_creation(self, sub_dataset_image_ids_file_path) -> list:
        """
        _sub_dataset_creation - This method creates the segmentation annotations for coco using
                                self._generate_segmentation_mask that uses the sub-classes inclusion tuple to keep only
                                the annotations that are relevant to the sub-classes selected when instantiating the class
            :param  sub_dataset_image_ids_file_path: The path to save the sub-dataset in for future loading
            :return:            All of the ids with enough pixel data after the sub-classing
        """
        print(
            'Creating sub-dataset , this will take a while but don\'t worry, it only runs once and caches the results')
        all_coco_image_ids = list(self.coco.imgs.keys())
        tbar = tqdm(all_coco_image_ids, desc='Generating sub-dataset image ids')
        sub_dataset_image_ids = []
        for i, img_id in enumerate(tbar):
            coco_target_annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]

            mask = self._generate_coco_segmentation_mask(coco_target_annotations, img_metadata['height'],
                                                         img_metadata['width'])

            # MAKE SURE THERE IS ENOUGH INPUT IN THE IMAGE (MORE THAN 1K PIXELS) AFTER SUB-CLASSES FILTRATION
            if (mask > 0).sum() > 1000:
                sub_dataset_image_ids.append(img_id)

            tbar.set_description('Processed images: {}/{}, generated {} qualified images'.
                                 format(i, len(all_coco_image_ids), len(sub_dataset_image_ids)))
        print('Number of images in sub-dataset: ', len(sub_dataset_image_ids))
        torch.save(sub_dataset_image_ids, sub_dataset_image_ids_file_path)
        return sub_dataset_image_ids
