import os

from pycocotools.coco import COCO
from super_gradients.common.abstractions.abstract_logger import get_logger
from torch.utils.data import Dataset
import random

import cv2
import numpy as np

from super_gradients.training.utils.detection_utils import get_yolox_datadir, random_affine, get_mosaic_coordinate, \
    adjust_box_anns
from super_gradients.training.utils.distributed_training_utils import get_local_rank

logger = get_logger(__name__)


class COCODetectionDatasetYolox(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
            self, img_size=(640, 640),
            mosaic=True, preproc=None,
            degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
            mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
            mosaic_prob=1.0, mixup_prob=1.0,
            data_dir=None,
            json_file="instances_train2017.json",
            name="images/train2017",
            cache=False,
            tight_box_rotation=False,
            transforms=[],
            *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__()
        self.imgs = None
        self.data_dir = data_dir
        self.input_dim = img_size
        self.preproc = preproc
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))  # duplicate
        self.tight_box_rotation = tight_box_rotation
        remove_useless_info(self.coco, self.tight_box_rotation)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        if cache:  # cache after merged
            self._cache_images()

        self.transforms = transforms

        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __getitem__(self, idx):
        sample = self.load_sample(idx)
        sample = self.apply_transforms(sample)
        return sample["image"], sample["target"], sample["info"], sample["id"]

        # if self.enable_mosaic and random.random() < self.mosaic_prob:
        #     mosaic_labels = []
        #     mosaic_labels_seg = []
        #     input_h, input_w = self.input_dim[0], self.input_dim[1]
        #
        #     # yc, xc = s, s  # mosaic center x, y
        #     yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        #     xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
        #
        #     # 3 additional image indices
        #     indices = [idx] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]
        #
        #     for i_mosaic, index in enumerate(indices):
        #         sample = self.load_sample(index)
        #         img, _labels, _labels_seg, img_id = sample["image"], sample["target"],  sample["target_seg"], sample["id"]
        #         h0, w0 = img.shape[:2]  # orig hw
        #         scale = min(1. * input_h / h0, 1. * input_w / w0)
        #         img = cv2.resize(
        #             img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
        #         )
        #         # generate output mosaic image
        #         (h, w, c) = img.shape[:3]
        #         if i_mosaic == 0:
        #             mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
        #
        #         # suffix l means large image, while s means small image in mosaic aug.
        #         (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
        #             mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
        #         )
        #
        #         mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
        #         padw, padh = l_x1 - s_x1, l_y1 - s_y1
        #
        #         labels = _labels.copy()
        #         labels_seg = _labels_seg.copy()
        #         # Normalized xywh to pixel xyxy format
        #         if _labels.size > 0:
        #             labels[:, 0] = scale * _labels[:, 0] + padw
        #             labels[:, 1] = scale * _labels[:, 1] + padh
        #             labels[:, 2] = scale * _labels[:, 2] + padw
        #             labels[:, 3] = scale * _labels[:, 3] + padh
        #
        #             labels_seg[:, ::2] = scale * labels_seg[:, ::2] + padw
        #             labels_seg[:, 1::2] = scale * labels_seg[:, 1::2] + padh
        #         mosaic_labels_seg.append(labels_seg)
        #         mosaic_labels.append(labels)
        #
        #     if len(mosaic_labels):
        #         mosaic_labels = np.concatenate(mosaic_labels, 0)
        #         np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
        #         np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
        #         np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
        #         np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
        #         mosaic_labels_seg = np.concatenate(mosaic_labels_seg, 0)
        #         np.clip(mosaic_labels_seg[:, ::2], 0, 2 * input_w, out=mosaic_labels_seg[:, ::2])
        #         np.clip(mosaic_labels_seg[:, 1::2], 0, 2 * input_h, out=mosaic_labels_seg[:, 1::2])
        #
        #     mosaic_img, mosaic_labels = random_affine(
        #         mosaic_img,
        #         mosaic_labels,
        #         mosaic_labels_seg,
        #         target_size=(input_w, input_h),
        #         degrees=self.degrees,
        #         translate=self.translate,
        #         scales=self.scale,
        #         shear=self.shear,
        #     )
        #
        #     # -----------------------------------------------------------------
        #     # CopyPaste: https://arxiv.org/abs/2012.07177
        #     # -----------------------------------------------------------------
        #     if (
        #             self.enable_mixup
        #             and not len(mosaic_labels) == 0
        #             and random.random() < self.mixup_prob
        #     ):
        #         mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
        #     mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
        #     img_info = (mix_img.shape[1], mix_img.shape[0])
        #
        #     # -----------------------------------------------------------------
        #     # img_info and img_id are not used for training.
        #     # They are also hard to be specified on a mosaic image.
        #     # -----------------------------------------------------------------
        #     return mix_img, padded_labels, img_info, img_id
        #
        # else:
        #     sample = self.load_sample(idx)
        #     img, label, img_info, img_id = sample["image"], sample["target"], sample["info"], sample["id"]
        #     img, label = self.preproc(img, label, self.input_dim)
        #     return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self.load_anno(cp_index)
        cp_sample = self.load_sample(cp_index)
        img, cp_labels = cp_sample["image"], cp_sample["target"]

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        num_seg_values = 98 if self.tight_box_rotation else 0
        res_seg = np.ones((num_objs, num_seg_values))
        res_seg.fill(np.nan)

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            if self.tight_box_rotation:
                seg_points = [j for i in obj.get("segmentation", []) for j in i]
                if seg_points:
                    seg_points_c = np.array(seg_points).reshape((-1, 2)).astype(np.int)
                    seg_points_convex = cv2.convexHull(seg_points_c).ravel()
                else:
                    seg_points_convex = []
                res_seg[ix, :len(seg_points_convex)] = seg_points_convex

        r = min(self.input_dim[0] / height, self.input_dim[1] / width)
        res[:, :4] *= r
        res_seg *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return res, res_seg, img_info, resized_info, os.path.join(self.data_dir, self.name, file_name)

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.input_dim[0]
        max_w = self.input_dim[1]
        cache_file = os.path.join(self.data_dir, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_sample(self, index):
        id_ = self.ids[index]
        res, res_seg, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        sample = {"image": img, "target": res.copy(), "target_seg": res_seg, "info": img_info, "id": np.array([id_])}

        return sample

    def load_image(self, index):
        file_name = self.annotations[index][4]

        img_file = os.path.join(file_name)

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def __del__(self):
        del self.imgs

    def load_anno(self, index):
        return self.annotations[index][0]

    def _get_random_non_empty_target_idx(self):
        target = []
        while len(target) == 0:
            idx = random.randint(0, len(self.ids) - 1)
            target = self.load_anno(idx)
        return idx

    def _load_random_samples(self, count, non_empty_targets_only=False):
        inds = [self._get_random_non_empty_target_idx() if non_empty_targets_only else random.randint(0, len(self.ids) - 1) for _ in range(count)]
        return [self.load_sample(ind) for ind in inds]


    def _load_additional_inputs_for_transform(self, sample, transform):
        additional_samples_count = transform.additional_samples_count if hasattr(transform, "additional_samples_count") else 0
        non_empty_targets = transform.non_empty_targets if hasattr(transform, "non_empty_targets") else False
        additional_samples = self._load_random_samples(additional_samples_count, non_empty_targets)
        sample["additional_samples"] = additional_samples

    def apply_transforms(self, sample: dict):
        for transform in self.transforms:
            self._load_additional_inputs_for_transform(sample, transform)
            sample = transform(sample)
        return sample



def remove_useless_info(coco, use_seg_info=False):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset and not use_seg_info:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)

