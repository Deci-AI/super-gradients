import os
from pycocotools.coco import COCO
from super_gradients.common.abstractions.abstract_logger import get_logger
from torch.utils.data import Dataset
import random
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

logger = get_logger(__name__)


class COCODetectionDataset(Dataset):
    """Detection dataset COCO implementation"""

    def __init__(
            self, img_size: tuple,
            data_dir: str = None,
            json_file: str = "instances_train2017.json",
            name: str = "images/train2017",
            cache: bool = False,
            cache_dir_path: str = None,
            tight_box_rotation: bool = False,
            transforms: list = [],
            with_crowd: bool = True
    ):
        """
        :param img_size: tuple, Image size (when loaded, before transforms)
        :param data_dir: str, root path to coco data.
        :param json_file: str, path to coco json file, that resides in data_dir/annotations/json_file.
        :param name: str, sub directory of data_dir containing the data.
        :param cache: bool, whether to cache images
        :param cache_dir_path: str, path to a directory that will be used for caching (with memmap).
        :param tight_box_rotation: bool, whether to use of segmentation maps convex hull
         as target_seg (see load_sample docs).
        :param transforms: list of transforms to apply sequentially on sample in __getitem__
        :param with_crowd: Add the crowd groundtruths to __getitem__
        """
        super().__init__()
        self.imgs = None
        self.data_dir = data_dir
        self.input_dim = img_size
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.with_crowd = with_crowd

        annotation_file_path = os.path.join(self.data_dir, "annotations", self.json_file)
        if not os.path.exists(annotation_file_path):
            raise ValueError("Could not find annotation file under " + str(annotation_file_path))
        self.coco = COCO(annotation_file_path)  # duplicate
        self.tight_box_rotation = tight_box_rotation
        remove_useless_info(self.coco, self.tight_box_rotation)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        if cache:  # cache after merged
            if cache_dir_path is None or not os.path.exists(cache_dir_path):
                raise ValueError("Must pass valid path through cache_dir_path when caching. Got " + str(cache_dir_path))
            self.cache_dir_path = cache_dir_path
            self._cache_images()

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.load_sample(idx)
        sample = self.apply_transforms(sample)
        if self.with_crowd:
            return sample["image"], sample["target"], sample["crowd_target"], sample["info"], sample["id"]
        else:
            return sample["image"], sample["target"], sample["info"], sample["id"]

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self._load_anno_from_ids(_ids) for _ids in tqdm(self.ids, desc="Loading annotations")]

    def _load_anno_from_ids(self, id_):
        """
        Load relevant information of a specific image

        :param id_: image id
        :return res:            Target Bboxes (detection)
        :return res_crowd:      Crowd target Bboxes (detection)
        :return res_seg:        Segmentation
        :return img_info:       Image (height, width)
        :return resized_info:   Resides image (height, width)
        :return img_path:       Path to the associated image
        """
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)])
        annotations = self.coco.loadAnns(anno_ids)

        cleaned_annotations = []
        for annotation in annotations:
            x1 = np.max((0, annotation["bbox"][0]))
            y1 = np.max((0, annotation["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, annotation["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, annotation["bbox"][3]))))
            if annotation["area"] > 0 and x2 >= x1 and y2 >= y1:
                annotation["clean_bbox"] = [x1, y1, x2, y2]
                cleaned_annotations.append(annotation)

        non_crowd_annotations = [annotation for annotation in cleaned_annotations if annotation["iscrowd"] == 0]

        res = np.zeros((len(non_crowd_annotations), 5))
        num_seg_values = 98 if self.tight_box_rotation else 0
        res_seg = np.ones((len(non_crowd_annotations), num_seg_values))
        res_seg.fill(np.nan)
        for ix, annotation in enumerate(non_crowd_annotations):
            cls = self.class_ids.index(annotation["category_id"])
            res[ix, 0:4] = annotation["clean_bbox"]
            res[ix, 4] = cls
            if self.tight_box_rotation:
                seg_points = [j for i in annotation.get("segmentation", []) for j in i]
                if seg_points:
                    seg_points_c = np.array(seg_points).reshape((-1, 2)).astype(np.int)
                    seg_points_convex = cv2.convexHull(seg_points_c).ravel()
                else:
                    seg_points_convex = []
                res_seg[ix, :len(seg_points_convex)] = seg_points_convex

        crowd_annotations = [annotation for annotation in cleaned_annotations if annotation["iscrowd"] == 1]

        res_crowd = np.zeros((len(crowd_annotations), 5))
        for ix, annotation in enumerate(crowd_annotations):
            cls = self.class_ids.index(annotation["category_id"])
            res_crowd[ix, 0:4] = annotation["clean_bbox"]
            res_crowd[ix, 4] = cls

        r = min(self.input_dim[0] / height, self.input_dim[1] / width)
        res[:, :4] *= r
        res_crowd[:, :4] *= r
        res_seg *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )
        img_path = os.path.join(self.data_dir, self.name, file_name)
        return res, res_crowd, res_seg, img_info, resized_info, img_path

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
        cache_file = os.path.join(self.cache_dir_path, f"img_resized_cache_{self.name.replace('/', '_')}.array")
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
        """
        Loads image at index, and resizes it to self.input_dim

        :param index: index to load the image from
        :return: resized_img
        """
        img = self.load_image(index)
        r = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_sample(self, index):
        """
        Loads sample at self.ids[index] as dictionary that holds:
            "image": Image resized to self.input_dim
            "target": Detection ground truth, np.array shaped (num_targets, 5), format is [class,x1,y1,x2,y2] with
                image coordinates.
            "target_seg": Segmentation map convex hull derived detection target.
            "info": Original shape (height,width).
            "id": COCO image id

        :param index: Sample index
        :return: sample as described above
        """
        id_ = self.ids[index]
        res, res_crowd, res_seg, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        sample = {"image": img, "target": res.copy(), "target_seg": res_seg, "info": img_info, "id": np.array([id_])}
        if self.with_crowd:
            sample["crowd_target"] = res_crowd.copy()
        return sample

    def load_image(self, index):
        """
        Loads image at index with its original resolution
        :param index: index in self.annotations
        :return: image (np.array)
        """
        file_name = self.annotations[index][5]

        img_file = os.path.join(file_name)

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def __del__(self):
        del self.imgs

    def _load_anno(self, index):
        return self.annotations[index][0]

    def _get_random_non_empty_target_idx(self):
        target = []
        while len(target) == 0:
            idx = random.randint(0, len(self.ids) - 1)
            target = self._load_anno(idx)
        return idx

    def _load_random_samples(self, count, non_empty_targets_only=False):
        inds = [
            self._get_random_non_empty_target_idx() if non_empty_targets_only else random.randint(0, len(self.ids) - 1)
            for _ in range(count)]
        return [self.load_sample(ind) for ind in inds]

    def _load_additional_inputs_for_transform(self, sample, transform):
        additional_samples_count = transform.additional_samples_count if hasattr(transform,
                                                                                 "additional_samples_count") else 0
        non_empty_targets = transform.non_empty_targets if hasattr(transform, "non_empty_targets") else False
        additional_samples = self._load_random_samples(additional_samples_count, non_empty_targets)
        sample["additional_samples"] = additional_samples

    def apply_transforms(self, sample: dict):
        """
        Applies self.transforms sequentially to sample

        If a transforms has the attribute 'additional_samples_count', additional samples will be loaded and stored in
         sample["additional_samples"] prior to applying it. Combining with the attribute "non_empty_targets" will load
         only additional samples with objects in them.

        :param sample: Sample to apply the transforms on to (loaded with self.load_sample)
        :return: Transformed sample
        """
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
