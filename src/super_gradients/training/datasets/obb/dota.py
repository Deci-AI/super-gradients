import multiprocessing
import random
import cv2
import numpy as np

from functools import partial
from pathlib import Path
from typing import Tuple, Iterable
from tqdm import tqdm

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.object_names import Processings
from super_gradients.common.registry import register_dataset
from super_gradients.dataset_interfaces import HasClassesInformation
from super_gradients.training.transforms import OBBDetectionCompose
from super_gradients.training.transforms.obb import OBBSample
from torch.utils.data import Dataset
from super_gradients.common.factories.transforms_factory import TransformsFactory

__all__ = ["DOTAOBBDataset"]


@register_dataset()
class DOTAOBBDataset(Dataset, HasClassesInformation):
    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        data_dir,
        transforms,
        class_names: Iterable[str],
        ignore_empty_annotations: bool = False,
        difficult_labels_are_crowd: bool = False,
        images_ext: str = ".jpg",
        images_subdir="images",
        ann_subdir="ann-obb",
    ):
        super().__init__()

        images_dir = Path(data_dir) / images_subdir
        ann_dir = Path(data_dir) / ann_subdir
        images, labels = self.find_images_and_labels(images_dir, ann_dir, images_ext)
        self.images = []
        self.coords = []
        self.classes = []
        self.difficult = []
        self.transforms = OBBDetectionCompose(transforms, load_sample_fn=self.load_random_sample)
        self.class_names = list(class_names)
        self.difficult_labels_are_crowd = difficult_labels_are_crowd

        class_names_to_index = {name: i for i, name in enumerate(self.class_names)}
        for image_path, label_path in tqdm(zip(images, labels), desc=f"Parsing annotations in {ann_dir}", total=len(images)):
            coords, classes, difficult = self.parse_annotation_file(label_path)
            if ignore_empty_annotations and len(coords) == 0:
                continue
            self.images.append(image_path)
            self.coords.append(coords)
            self.classes.append(np.array([class_names_to_index[c] for c in classes], dtype=int))
            self.difficult.append(difficult)

    def __len__(self):
        return len(self.images)

    def load_random_sample(self) -> OBBSample:
        num_samples = len(self)
        random_index = random.randrange(0, num_samples)
        return self.load_sample(random_index)

    def load_sample(self, index) -> OBBSample:
        image = cv2.imread(str(self.images[index]))
        coords = self.coords[index]
        classes = self.classes[index]
        difficult = self.difficult[index]
        cxcywhr = np.array([self.poly_to_rbox(poly) for poly in coords], dtype=np.float32)

        is_crowd = difficult.reshape(-1) if self.difficult_labels_are_crowd else np.zeros_like(difficult, dtype=bool)
        sample = OBBSample(
            image=image,
            rboxes_cxcywhr=cxcywhr.reshape(-1, 5),
            labels=classes.reshape(-1),
            is_crowd=is_crowd,
        )
        return sample

    def __getitem__(self, index) -> OBBSample:
        sample = self.load_sample(index)
        sample = self.transforms.apply_to_sample(sample)
        return sample

    def get_sample_classes_information(self, index) -> np.ndarray:
        """
        Returns a histogram of length `num_classes` with class occurrences at that index.
        """
        return np.bincount(self.classes[index], minlength=len(self.class_names))

    def get_dataset_classes_information(self) -> np.ndarray:
        """
        Returns a matrix of shape (dataset_length, num_classes). Each row `i` is histogram of length `num_classes` with class occurrences for sample `i`.
        Example implementation, assuming __len__: `np.vstack([self.get_sample_classes_information(i) for i in range(len(self))])`
        """
        m = np.zeros((len(self), len(self.class_names)), dtype=int)
        for i in range(len(self)):
            m[i] = self.get_sample_classes_information(i)
        return m

    def get_dataset_preprocessing_params(self):
        """
        Return any hardcoded preprocessing + adaptation for PIL.Image image reading (RGB).
         image_processor as returned as list of dicts to be resolved by processing factory.
        :return:
        """
        pipeline = [Processings.ReverseImageChannels]
        for t in self.transforms:
            pipeline += t.get_equivalent_preprocessing()
        params = dict(
            class_names=self.class_names,
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            iou=0.65,
            conf=0.5,
        )
        return params

    @classmethod
    def poly_to_rbox(cls, poly):
        """
        Convert polygon to rotated bounding box
        :param poly: Input polygon in [N,2] format
        :return: Rotated box in CXCYWHR format
        """
        hull = cv2.convexHull(np.reshape(poly, [-1, 2]))
        rect = cv2.minAreaRect(hull)
        cx, cy = rect[0]
        w, h = rect[1]
        angle = rect[2]
        if angle == 0:
            w, h = h, w
            angle -= 90
        return cx, cy, w, h, np.deg2rad(angle)

    @classmethod
    def find_images_and_labels(cls, images_dir, ann_dir, images_ext):
        images_dir = Path(images_dir)
        ann_dir = Path(ann_dir)

        images = list(images_dir.glob(f"*{images_ext}"))
        labels = list(sorted(ann_dir.glob("*.txt")))

        if len(images) != len(labels):
            raise ValueError(f"Number of images and labels do not match. There are {len(images)} images and {len(labels)} labels.")

        images = []
        for label_path in labels:
            image_path = images_dir / (label_path.stem + images_ext)
            if not image_path.exists():
                raise ValueError(f"Image {image_path} does not exist")
            images.append(image_path)
        return images, labels

    @classmethod
    def parse_annotation_file(cls, annotation_file: Path):
        with open(annotation_file, "r") as f:
            lines = f.readlines()

        coords = []
        classes = []
        difficult = []

        for line in lines:
            parts = line.strip().split(" ")
            if len(parts) != 10:
                raise ValueError(f"Invalid number of parts in line: {line}")

            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            coords.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            classes.append(parts[8])
            difficult.append(int(parts[9]))

        return np.array(coords, dtype=np.float32).reshape(-1, 4, 2), np.array(classes, dtype=np.object_), np.array(difficult, dtype=int)

    @classmethod
    def chip_image(cls, img, coords, classes, difficult, tile_size: Tuple[int, int], tile_step: Tuple[int, int], min_visibility: float, min_area: int):
        """
        Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip.

        :param img: the image to be chipped in array format
        :param coords: an (N,4,2) array of oriented box coordinates for that image
        :param classes: an (N,1) array of classes for each bounding box
        :param tile_size: an (W,H) tuple indicating width and height of chips

        Output:
            An image array of shape (M,W,H,C), where M is the number of chips,
            W and H are the dimensions of the image, and C is the number of color
            channels.  Also returns boxes and classes dictionaries for each corresponding chip.
        """
        height, width, _ = img.shape

        tile_size_width, tile_size_height = tile_size
        tile_step_width, tile_step_height = tile_step

        total_images = []
        total_boxes = []
        total_classes = []
        total_difficult = []

        start_x = 0
        end_x = start_x + tile_size_width

        all_areas = np.array(list(cv2.contourArea(cv2.convexHull(poly)) for poly in coords), dtype=np.float32)

        centers = np.mean(coords, axis=1)  # [N,2]

        while start_x < width:
            start_y = 0
            end_y = start_y + tile_size_height
            while start_y < height:
                chip = img[start_y:end_y, start_x:end_x, :3]

                # Skipping thin strips that are not useful
                # For instance, if image is 1030px wide and our tile size is 1024, that would end up with
                # two tiles of [1024, 1024] and [1024, 6] which is not useful at all
                if chip.shape[0] > 8 or chip.shape[1] > 8:

                    # Filter out boxes that whose bounding box is definitely not in the chip
                    offset = np.array([start_x, start_y], dtype=np.float32)
                    boxes_with_offset = coords - offset.reshape(1, 1, 2)
                    centers_with_offset = centers - offset.reshape(1, 2)

                    cond1 = (centers_with_offset >= 0).all(axis=1)
                    cond2 = (centers_with_offset[:, 0] < chip.shape[1]) & (centers_with_offset[:, 1] < chip.shape[0])
                    rboxes_inside_chip = cond1 & cond2

                    visible_coords = boxes_with_offset[rboxes_inside_chip]
                    visible_classes = classes[rboxes_inside_chip]
                    visible_difficult = difficult[rboxes_inside_chip]
                    visible_areas = all_areas[rboxes_inside_chip]

                    out_clipped = np.stack(
                        (
                            np.clip(visible_coords[:, :, 0], 0, chip.shape[1]),
                            np.clip(visible_coords[:, :, 1], 0, chip.shape[0]),
                        ),
                        axis=2,
                    )
                    areas_clipped = np.array(list(cv2.contourArea(cv2.convexHull(c)) for c in out_clipped), dtype=np.float32)

                    visibility_fraction = areas_clipped / (visible_areas + 1e-6)
                    visibility_mask = visibility_fraction >= min_visibility
                    min_area_mask = areas_clipped >= min_area

                    visible_coords = visible_coords[visibility_mask & min_area_mask]
                    visible_classes = visible_classes[visibility_mask & min_area_mask]
                    visible_difficult = visible_difficult[visibility_mask & min_area_mask]

                    total_boxes.append(visible_coords)
                    total_classes.append(visible_classes)
                    total_difficult.append(visible_difficult)

                    if chip.shape[0] < tile_size_height or chip.shape[1] < tile_size_width:
                        chip = cv2.copyMakeBorder(
                            chip,
                            top=0,
                            left=0,
                            bottom=tile_size_height - chip.shape[0],
                            right=tile_size_width - chip.shape[1],
                            value=0,
                            borderType=cv2.BORDER_CONSTANT,
                        )
                    total_images.append(chip)

                start_y += tile_step_height
                end_y += tile_step_height

            start_x += tile_step_width
            end_x += tile_step_width

        return total_images, total_boxes, total_classes, total_difficult

    @classmethod
    def slice_dataset_into_tiles(
        cls,
        data_dir,
        output_dir,
        ann_subdir_name,
        tile_size: int,
        tile_step: int,
        scale_factors: Tuple,
        min_visibility,
        min_area,
        num_workers: int,
        output_image_ext=".jpg",
    ):
        data_dir = Path(data_dir)
        input_images_dir = data_dir / "images"
        input_ann_dir = data_dir / ann_subdir_name
        images, labels = cls.find_images_and_labels(input_images_dir, input_ann_dir, ".png")

        output_dir = Path(output_dir)
        output_images_dir = output_dir / "images"
        output_ann_dir = output_dir / ann_subdir_name

        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_ann_dir.mkdir(parents=True, exist_ok=True)

        with multiprocessing.Pool(num_workers) as wp:
            payload = [(image_path, ann_path, scale) for image_path, ann_path in zip(images, labels) for scale in scale_factors]

            worker_fn = partial(
                cls._worker_fn,
                tile_size=tile_size,
                tile_step=tile_step,
                min_visibility=min_visibility,
                min_area=min_area,
                output_images_dir=output_images_dir,
                output_ann_dir=output_ann_dir,
                output_image_ext=output_image_ext,
            )
            for _ in tqdm(wp.imap_unordered(worker_fn, payload), total=len(payload)):
                pass

    @classmethod
    def _worker_fn(cls, args, tile_size, tile_step, min_visibility, min_area, output_images_dir, output_ann_dir, output_image_ext):
        image_path, ann_path, scale = args
        image = cv2.imread(str(image_path))
        coords, classes, difficult = cls.parse_annotation_file(ann_path)
        scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        image_tiles, total_boxes, total_classes, total_difficult = cls.chip_image(
            scaled_image,
            coords * scale,
            classes,
            difficult,
            tile_size=(tile_size, tile_size),
            tile_step=(tile_step, tile_step),
            min_visibility=min_visibility,
            min_area=min_area,
        )
        num_tiles = len(image_tiles)

        for i in range(num_tiles):
            tile_image = image_tiles[i]
            tile_boxes = total_boxes[i]
            tile_classes = total_classes[i]
            tile_difficult = total_difficult[i]

            tile_image_path = output_images_dir / f"{ann_path.stem}_{scale:.3f}_{i:06d}{output_image_ext}"
            tile_label_path = output_ann_dir / f"{ann_path.stem}_{scale:.3f}_{i:06d}.txt"

            with tile_label_path.open("w") as f:
                for poly, category, diff in zip(tile_boxes, tile_classes, tile_difficult):
                    f.write(
                        f"{poly[0, 0]:.2f} {poly[0, 1]:.2f} {poly[1, 0]:.2f} {poly[1, 1]:.2f} {poly[2, 0]:.2f} {poly[2, 1]:.2f} {poly[3, 0]:.2f} {poly[3, 1]:.2f} {category} {diff}\n"  # noqa
                    )

            cv2.imwrite(str(tile_image_path), tile_image)
