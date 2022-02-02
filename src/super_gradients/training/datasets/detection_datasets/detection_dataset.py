import os
import math
import random
import cv2
import numpy as np
import torch
from typing import Callable
from tqdm import tqdm
from PIL import Image, ExifTags
from super_gradients.training.datasets.sg_dataset import ListDataset
from super_gradients.training.utils.detection_utils import convert_xyxy_bbox_to_xywh
from super_gradients.training.utils.utils import get_param
# PREVENTS THE cv2 DEADLOCK
cv2.setNumThreads(0)

# GET ORIENTATION EXIF TAG
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


class DetectionDataSet(ListDataset):
    def __init__(self, root: str, list_file: str, img_size: int = 416, batch_size: int = 16, augment: bool = False,
                 dataset_hyper_params: dict = None, cache_labels: bool = False, cache_images: bool = False,
                 sample_loading_method: str = 'default', collate_fn: Callable = None, target_extension: str = '.txt',
                 labels_offset: int = 0, class_inclusion_list=None, all_classes_list=None):
        """
        DetectionDataSet
            :param root:                    Root folder of the Data Set
            :param list_file:
            :param img_size:                Image size of the Model that uses this Data Set
            :param batch_size:              Batch Size
            :param augment:                 True / False flag to allow Augmentation
            :param dataset_hyper_params:    Any hyper params required for the data set
            :param cache_labels:            "Caches" the labels -> Pre-Loads to memory as a list
                IMPORTANT NOTE: CURRENTLY OBJECTLESS IMAGES ARE DISCARDED ONLY WHEN THIS IS SET. THEREFORE A GOOD PRACTICE
                WHEN USING SUBCLASSING IS TO SET THIS PARAMETER TO TRUE.
            :param cache_images:            "Caches" the images -> Pre-Loads to memory as a list
            :param sample_loading_method:   default - Normal Training... No Special Augmentation
                                            mosaic -  Used *ONLY* for training improvement, creates a new image that is
                                                      comprised of 4 randomly selected images that are located in random
                                                      sizes in 4 different parts of the image (Extreme Augmentation)
                                            rectangular - Used mainly for inference, it letterboxes the image
                                                          for the expected image size of the model
            :param labels_offset:           offset value to add to the labels (class numbers)
            :param all_classes_list: list(str) containing all the class names.
            :param class_inclusion_list: list(str) containing the subclass names or None when subclassing is disabled.
        """
        self.dataset_hyperparams = dataset_hyper_params
        self.cache_labels = cache_labels
        self.cache_images = cache_images
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.batch_index = None
        self.total_batches_num = None
        self.sample_loading_method = sample_loading_method
        self.labels_offset = labels_offset

        self.class_inclusion_list = class_inclusion_list
        self.all_classes_list = all_classes_list
        self.mixup_prob = get_param(self.dataset_hyperparams, "mixup", 0)

        super(DetectionDataSet, self).__init__(root=root, file=list_file, target_extension=target_extension,
                                               collate_fn=collate_fn, sample_loader=self.sample_loader,
                                               sample_transform=self.sample_transform, target_loader=self.target_loader,
                                               target_transform=self.target_transform)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        label_path = self.label_files[index]

        if self.sample_loading_method == 'mosaic' and self.augment:
            # LOAD 4 IMAGES AT A TIME INTO A MOSAIC (ONLY DURING TRAINING)
            img, labels = self.load_mosaic(index)
            # MixUp augmentation
            if random.random() < self.mixup_prob:
                img, labels = self.mixup(img, labels, *self.load_mosaic(random.randint(0, len(self.img_files) - 1)))

        else:
            # LOAD A SINGLE IMAGE
            img_path = self.img_files[index]
            img = self.sample_loader(img_path)
            img = self.sample_transform(img)

            # LETTERBOX
            h, w = img.shape[:2]
            shape = self.batch_shapes[
                self.batch_index[index]] if self.sample_loading_method == 'rectangular' else self.img_size
            img, ratio, pad = self.letterbox(img, shape, auto=False, scaleup=self.augment)

            # LOAD LABELS
            if self.cache_labels:
                labels = self.labels[index]
            else:
                labels = self.target_loader(label_path, self.class_inclusion_list, self.all_classes_list)

            labels = self.target_transform(labels, ratio, w, h, pad)

        if self.augment:
            # AUGMENT IMAGESPACE
            if not self.sample_loading_method == 'mosaic':
                img, labels = self.random_perspective(img, labels,
                                                      degrees=self.dataset_hyperparams['degrees'],
                                                      translate=self.dataset_hyperparams['translate'],
                                                      scale=self.dataset_hyperparams['scale'],
                                                      shear=self.dataset_hyperparams['shear'])

            # AUGMENT COLORSPACE
            img = self.augment_hsv(img, hgain=self.dataset_hyperparams['hsv_h'],
                                   sgain=self.dataset_hyperparams['hsv_s'],
                                   vgain=self.dataset_hyperparams['hsv_v'])

        if len(labels):
            # CONVERT XYXY TO XYWH - CHANGE THE BBOXES PARAMS
            labels[:, 1:5] = convert_xyxy_bbox_to_xywh(labels[:, 1:5])

            # NORMALIZE COORDINATES 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # HEIGHT
            labels[:, [1, 3]] /= img.shape[1]  # WIDTH

        if self.augment:
            # AUGMENT RANDOM UP-DOWN FLIP
            img, labels = self._augment_random_flip(img, labels)

        labels_out = torch.zeros((len(labels), 6))

        labels[:, 0] += self.labels_offset
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = self.sample_post_process(img)

        return img, labels_out

    @staticmethod
    def mixup(im, labels, im2, labels2):
        # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
        # https://github.com/ultralytics/yolov5
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        im = (im * r + im2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        return im, labels

    @staticmethod
    def sample_post_process(image):
        """
        sample_post_process - Normalizes and orders the image to be 3 x img_size x img_size
            :param image:
            :return:
        """
        # CONVERT BGR to RGB, to 3xIMAGE_SIZExIMAGE_SIZE
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        # 0 - 255 TO 0.0 - 1.0
        return torch.from_numpy(image) / 255.0

    def _generate_samples_and_targets(self):
        """
        _generate_samples_and_targets
        """
        self.classes = self.class_inclusion_list or self.all_classes_list

        if not self.samples_targets_tuples_list:
            list_file = open(self.root + os.path.sep + self.list_file_path, "r", encoding="utf-8")

            self.img_files = [x.replace('/', os.sep) for x in list_file.read().splitlines()
                              if os.path.splitext(x)[-1].lower() in self.extensions]
            self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], self.target_extension)
                                for x in self.img_files]
        else:
            self.img_files, self.label_files = map(list, zip(*self.samples_targets_tuples_list))

        samples_len = len(self.img_files)
        self.batch_index = np.floor(np.arange(samples_len) / self.batch_size).astype(np.int)
        self.total_batches_num = self.batch_index[-1] + 1

        # RECTANGULAR TRAINING
        if self.sample_loading_method == 'rectangular':
            self._rectangular_loading(samples_len=samples_len)

        # PRELOAD LABELS (REQUIRED FOR WEIGHTED CE TRAINING)
        self.imgs = [None] * samples_len
        self.labels = [None] * samples_len

        if self.cache_labels:
            self.labels = [np.zeros((0, 5))] * samples_len
            pbar = tqdm(self.label_files, desc='Caching labels')
            missing_labels, found_labels, duplicate_labels = 0, 0, 0
            image_indices_to_keep = []

            for i, file in enumerate(pbar):
                labels = self.target_loader(file, self.class_inclusion_list, self.all_classes_list)
                if labels is None:
                    missing_labels += 1
                    continue

                self.labels[i] = labels
                found_labels += 1
                image_indices_to_keep.append(i)

                pbar.desc = 'Caching labels (%g found, %g missing, %g duplicate, for %g images)' % (
                    found_labels, missing_labels, duplicate_labels, samples_len)
            assert found_labels > 0, 'No labels found.'

            image_indices_to_keep = set(image_indices_to_keep)
            #  REMOVE THE IRRELEVANT ENTRIES FROM THE DATA
            self.img_files = [e for i, e in enumerate(self.img_files) if i in image_indices_to_keep]
            self.label_files = [e for i, e in enumerate(self.label_files) if i in image_indices_to_keep]
            self.imgs = [e for i, e in enumerate(self.imgs) if i in image_indices_to_keep]
            self.labels = [e for i, e in enumerate(self.labels) if i in image_indices_to_keep]

        # CACHE IMAGES INTO MEMORY FOR FASTER TRAINING (WARNING: LARGE DATASETS MAY EXCEED SYSTEM RAM)
        if self.cache_images:
            cached_images_mem_in_gb = 0
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            for i in pbar:
                img_path = self.img_files[i]
                img = self.sample_loader(img_path)
                img = self.sample_transform(img)
                self.imgs[i] = img
                cached_images_mem_in_gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (cached_images_mem_in_gb / 1E9)

    def _rectangular_loading(self, samples_len, pad: float = 0.5):
        """

        :param samples_len:
        :return:
        """
        shapes_file_path = self.root + self.list_file_path.replace('.txt', '.shapes')
        try:
            with open(shapes_file_path, 'r') as shapes_file:
                image_shapes = [row.split() for row in shapes_file.read().splitlines()]
                assert len(image_shapes) == samples_len, 'Shapefile out of sync'
        except Exception as ex:
            print(ex)
            image_shapes = [self.exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
            # np.savetxt(shapes_file_path, image_shapes, fmt='%g')
            # TODO: we are not writing the shapes into a file since this is risky. the files list might change
            # TODO: and then the shapes in the .shapes file will be wrong

        # SORT BY ASPECT RATIO
        image_shapes = np.array(image_shapes, dtype=np.float64)
        aspect_ratio = image_shapes[:, 1] / image_shapes[:, 0]
        sorted_indices = aspect_ratio.argsort()

        self.img_files = [self.img_files[i] for i in sorted_indices]
        self.label_files = [self.label_files[i] for i in sorted_indices]
        self.shapes = image_shapes[sorted_indices]
        aspect_ratio = aspect_ratio[sorted_indices]

        # SET IMAGE SHAPES
        shapes = [[1, 1]] * self.total_batches_num
        for i in range(self.total_batches_num):
            ari = aspect_ratio[self.batch_index == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / 32. + pad).astype(np.int) * 32

    @staticmethod
    def _augment_random_flip(image, labels) -> tuple:
        """
        _augment_random_flip
            :param image:
            :param labels:
            :return:
        """
        # RANDOM LEFT-RIGHT FLIP
        if random.random() < 0.5:
            image = np.fliplr(image)
            if len(labels):
                labels[:, 1] = 1 - labels[:, 1]

        return image, labels

    @staticmethod
    def sample_loader(sample_path: str):
        """
        sample_loader - Loads a coco dataset image from path
            :param sample_path:
            :return:
        """
        image = None
        if sample_path is not None:
            # BGR
            try:
                image = cv2.imread(sample_path)
                if image is None:
                    print('Image Not Found: ' + sample_path)

            except Exception as ex:
                print('Caught Exception trying to to open ' + sample_path + str(ex))

        return image

    def sample_transform(self, image):
        """
        sample_transform
            :param image:
            :return:
        """
        # RESIZE IMAGE TO IMG_SIZE
        resize = self.img_size / max(image.shape)

        # ALWAYS RESIZE DOWN, ONLY RESIZE UP IF TRAINING WITH AUGMENTATION
        if resize != 1:
            h, w = image.shape[:2]
            # resize in test with AREA interpolation - according to https://github.com/ultralytics/yolov5
            interp = cv2.INTER_AREA if resize < 1 and not self.augment else cv2.INTER_LINEAR
            return cv2.resize(image, (int(w * resize), int(h * resize)), interpolation=interp)

        return image

    @staticmethod
    def target_loader(target_path: str, class_inclusion_list=None, all_classes_list=None):
        """
        coco_target_loader
            @param target_path: str, path to target.
            @param all_classes_list: list(str) containing all the class names or None when subclassing is disabled.
            @param class_inclusion_list: list(str) containing the subclass names or None when subclassing is disabled.
        """
        target = None
        if os.path.isfile(target_path):
            try:
                with open(target_path, 'r') as targets_file:
                    target = np.array([x.split() for x in targets_file.read().splitlines()], dtype=np.float32)

                if target.shape[0]:
                    assert target.shape[1] == 5, '> 5 label columns: %s' % target_path
                    assert (target >= 0).all(), 'negative labels: %s' % target_path
                    assert (target[:,
                            1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % target_path

            except Exception as ex:
                print('Caught Exception trying to to open ' + target_path + str(ex))

        # SUBCLASSING
        if class_inclusion_list is not None and target is not None:
            # FILTER THE EXCLUDED CLASSES
            target = np.array(list(filter(lambda lbl: all_classes_list[int(lbl[0])] in class_inclusion_list, target)))
            if len(target):
                # MAP THE INCLUDED CLASSES TO THEIR NEW INDICES IN THE CLASSES INCLUSION LIST
                target[:, 0] = np.array(
                    list(map(lambda lbl: class_inclusion_list.index(all_classes_list[int(lbl[0])]), target)))
            else:
                # HANDLING THE CASE WHEN THERE ARE NO OBJECTS LEFT AFTER FILTERING
                target = None
        return target

    @staticmethod
    def target_transform(target, ratio, w, h, pad=None):
        """
        target_transform
            :param target:
            :param ratio:
            :param w:
            :param h:
            :param pad:
            :return:
        """
        if target is None:
            return np.zeros((0, 5), dtype=np.float32)

        # HANDLE EDGE CASE
        if not target.size > 0:
            labels = np.zeros((0, 5), dtype=np.float32)
            return labels, 0, 0, (0, 0), (0, 0)

        # NORMALIZED xywh TO PIXEL xyxy FORMAT
        labels = target.copy()
        labels[:, 1] = ratio[0] * w * (target[:, 1] - target[:, 3] / 2) + pad[0]
        labels[:, 2] = ratio[1] * h * (target[:, 2] - target[:, 4] / 2) + pad[1]
        labels[:, 3] = ratio[0] * w * (target[:, 1] + target[:, 3] / 2) + pad[0]
        labels[:, 4] = ratio[1] * h * (target[:, 2] + target[:, 4] / 2) + pad[1]

        return labels

    @staticmethod
    def exif_size(img):
        """
        exif_size
            :param img:
            :return:
        """
        # RETURNS EXIF-CORRECTED PIL SIZE (width, height)
        image_size = img.size
        try:
            exif_data = img._getexif()
            if exif_data is not None:
                rotation = dict(exif_data.items())[orientation]
                # ROTATION 270
                if rotation == 6:
                    image_size = (image_size[1], image_size[0])
                # ROTATION 90
                elif rotation == 8:
                    image_size = (image_size[1], image_size[0])
        except Exception as ex:
            print('Caught Exception trying to rotate: ' + str(img) + str(ex))

        return image_size

    @staticmethod
    def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
        """
        augment_hsv
            :param img:
            :param hgain:
            :param sgain:
            :param vgain:
        :return:
        """
        x = (np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1).astype(np.float32)  # random gains
        img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img

    @staticmethod
    def letterbox(img, new_shape=(416, 416), color=(128, 128, 128), auto=True, scaleFill=False, scaleup=True,
                  interp=cv2.INTER_AREA) -> tuple:
        """
        letterbox - Resizes image to a 32-pixel-multiple rectangle
        :param img:
        :param new_shape:
        :param color:
        :param auto:
        :param scaleFill:
        :param scaleup:
        :param interp:
        :return:
        """
        # CURRENT IMAGE SHAPE [HEIGHT, WIDTH]
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # ONLY SCALE DOWN, DO NOT SCALE UP (FOR BETTER TEST MAP)
        scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            scale_ratio = min(scale_ratio, 1.0)

        ratio = scale_ratio, scale_ratio
        unpadded_output_shape = int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio))
        dw, dh = new_shape[1] - unpadded_output_shape[0], new_shape[0] - unpadded_output_shape[1]

        # MINIMUM RECTANGLE - LEAVES BLANK SPACES IN THE IMAGE
        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)

        # STRETCH - STRETCHES THE IMAGE FOR THE DESIRED RESOLUTION
        elif scaleFill:
            dw, dh = 0.0, 0.0
            unpadded_output_shape = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

        dw /= 2
        dh /= 2

        # INTER_AREA IS BETTER, INTER_LINEAR IS FASTER
        if shape[::-1] != unpadded_output_shape:
            img = cv2.resize(img, unpadded_output_shape, interpolation=interp)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # ADDS BORDER
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

    def random_perspective(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0,
                           perspective=0):
        """
        random images and labels using a perspective transform
        """
        height = img.shape[0] + border * 2  # shape(h,w,c)
        width = img.shape[1] + border * 2

        # CENTER
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # PERSPECTIVE
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # ROTATION AND SCALE
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # A += RANDOM.CHOICE([-180, -90, 0, 90])  # ADD 90DEG ROTATIONS TO SMALL ROTATIONS
        s = random.uniform(1 - scale, 1 + scale)
        # S = 2 ** RANDOM.UNIFORM(-SCALE, SCALE)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # SHEAR
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # TRANSLATION
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # COMBINED ROTATION MATRIX
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border != 0) or (border != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # TRANSFORM LABEL COORDINATES
        num_targets = len(targets)
        if num_targets:
            # WARP POINTS
            xy = np.ones((num_targets * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(num_targets * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(num_targets, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(num_targets, 8)

            # CREATE NEW BOXES
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, num_targets).T

            # CLIP BOXES
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # FILTER CANDIDATES
            i = self.box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return img, targets

    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
        """
        compute candidate boxes
            :param box1:        before augment
            :param box2:        after augment
            :param wh_thr:      wh_thr (pixels)
            :param ar_thr:      aspect_ratio_thr
            :param area_thr:    area_ratio
        :return:
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

    def load_mosaic(self, index):
        """
       load_mosaic - Load images in mosaic format to improve noise handling while training
           :param index:
           :return:
        """
        mosaic_labels = []
        image_size = self.img_size

        # MOSAIC CENTER X, Y
        mosaic_center_x, mosaic_center_y = [int(random.uniform(image_size * 0.5, image_size * 1.5)) for _ in range(2)]

        # BASE IMAGE WITH 4 TILES
        mosaic_image = np.zeros((image_size * 2, image_size * 2, 3), dtype=np.uint8) + 128

        # 3 ADDITIONAL RANDOM IMAGE INDICES
        indices = [index] + [random.randint(0, len(self.label_files) - 1) for _ in range(3)]
        for img_index, index in enumerate(indices):
            if self.cache_images:
                img = self.imgs[index]
            else:
                img_path = self.img_files[index]
                img = self.sample_loader(img_path)
                img = self.sample_transform(img)

            h, w = img.shape[:2]

            mosaic_image, padw, padh = self._place_image_in_mosaic(mosaic_image, img, img_index, mosaic_center_x,
                                                                   mosaic_center_y, w, h)

            # LOAD LABELS
            if self.cache_labels:
                labels = self.labels[index]
            else:
                label_path = self.label_files[index]
                labels = self.target_loader(label_path, self.class_inclusion_list, self.all_classes_list)

            labels = self.target_transform(labels, ratio=(1, 1), w=w, h=h, pad=(padw, padh))
            mosaic_labels.append(labels)

        # Concat/clip labels
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 1:], 0, 2 * image_size, out=mosaic_labels[:, 1:])  # use with random_affine

        # AUGMENT AND REMOVE THE BORDER
        mosaic_image, mosaic_labels = self.random_perspective(mosaic_image, mosaic_labels,
                                                              degrees=self.dataset_hyperparams['degrees'],
                                                              translate=self.dataset_hyperparams['translate'],
                                                              scale=self.dataset_hyperparams['scale'],
                                                              shear=self.dataset_hyperparams['shear'],
                                                              border=-image_size // 2)

        return mosaic_image, mosaic_labels

    def _place_image_in_mosaic(self, mosaic_image, image, image_index, center_x, center_y, w, h):
        """
        _place_image_in_mosaic
            :param mosaic_image:
            :param image:
            :param image_index:
            :param center_x:
            :param center_y:
            :param w:
            :param h:
            :return:
        """
        image_size = self.img_size

        # PLACE IMG IN IMG4
        if image_index == 0:  # TOP LEFT
            x1a, y1a, x2a, y2a = max(center_x - w, 0), max(center_y - h, 0), center_x, center_y
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif image_index == 1:  # TOP RIGHT
            x1a, y1a, x2a, y2a = center_x, max(center_y - h, 0), min(center_x + w, image_size * 2), center_y
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif image_index == 2:  # BOTTOM LEFT
            x1a, y1a, x2a, y2a = max(center_x - w, 0), center_y, center_x, min(image_size * 2, center_y + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(center_x, w), min(y2a - y1a, h)
        elif image_index == 3:  # BOTTOM RIGHT
            x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + w, image_size * 2), min(image_size * 2,
                                                                                            center_y + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        return mosaic_image, padw, padh
